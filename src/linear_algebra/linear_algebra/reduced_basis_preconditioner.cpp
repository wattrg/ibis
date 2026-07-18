#include <linear_algebra/reduced_basis_preconditioner.h>
#include "impl/Kokkos_CheckUsage.hpp"
#include "linear_algebra/dense_linear_algebra.h"
#include "util/conserved_quantities.h"

template <typename MemModel>
ReducedBasisPreconditioner<MemModel>::ReducedBasisPreconditioner(std::shared_ptr<LinearSystem> system,
                                                                 int nb) {
    system_ = system;
    nb_max_ = nb;
    nb_ = -1; // start at minus 1, because the first "update_basis" call happens before the first solution
    n_vars_ = system_->num_vars();
    W_ = Ibis::Matrix<Ibis::real>("RBP::W", n_vars_, nb);
    D_dev_ = Ibis::Matrix<Ibis::real>("RBP::D", nb, nb);
    V_ = Ibis::Matrix<Ibis::real>("RBP::V", n_vars_, nb);
    H_ = Ibis::Matrix<Ibis::real>("RBP::H", nb, nb);
    H_inv_ = Ibis::Matrix<Ibis::real>("RBP::H_inv", nb, nb);
    w_ = Ibis::Vector<Ibis::real>("RBP::w", nb);
    p_ = Ibis::Vector<Ibis::real>("RBP::p", nb);
    d_ = Ibis::Vector<Ibis::real>("RBP::d", n_vars_);
    z_ = Ibis::Vector<Ibis::real>("RBP::z", n_vars_);
    s_ = Ibis::Vector<Ibis::real>("RBP::s", n_vars_);
    vec_tmp_ = Ibis::Vector<Ibis::real>("RBP::vec_tmp", n_vars_);
    res_tmp_ = Ibis::Vector<Ibis::real>("RBP::res_tmp", n_vars_);
    M_ = Ibis::Vector<Ibis::real>("RBP::M", n_vars_);
}

template <typename MemModel>
void ReducedBasisPreconditioner<MemModel>::update_basis(Ibis::Vector<Ibis::real>& sol) {
    if (nb_ == -1) {
        // The provided vector is not a previous solution, just the initial guess
        // so we will ignore it.
        nb_++;
        return;
    }
    if (nb_ < nb_max_) {
        // we haven't filled the reduced basis yet
        // so we can just add the new vector
        auto W_col = W_.column(nb_);
        W_col.deep_copy_layout(sol);
    } else {
        // Our reduced basis is full, so shuffle previous solutions along by 1 column
        // to make room for the new one, and discard the oldest solution
        for (int i = 1; i < nb_; i++) {
            auto W_previous = W_.column(i - 1);
            auto W_current = W_.column(i);
            W_previous.deep_copy_space(W_current);

            auto V_previous = V_.column(i - 1);
            auto V_current = V_.column(i);
            V_previous.deep_copy_space(V_current);
        }

        // add the new basis vector
        W_.column(nb_ - 1).deep_copy_layout(sol);
    }
    nb_ = Ibis::min(nb_ + 1, nb_max_);
    system_->leading_diagonal_term(M_);

    // V = AW - MW
    auto M = M_;
    for (size_t n = 0; n < nb_; n++) {
        auto Wn = W_.column(n);
        vec_tmp_.deep_copy_layout(Wn);
        system_->matrix_vector_product(vec_tmp_, res_tmp_);
        auto Vn = V_.column(n);
        auto res_tmp = res_tmp_;
        Kokkos::parallel_for("RBP::V", n_vars_, KOKKOS_LAMBDA(const int i){
            Vn(i) = res_tmp(i) - M(i) * Wn(i);   
        });
    }

    auto W_sub = W_.columns(0, nb_);
    auto D_sub = D_dev_.sub_matrix(0, nb_, 0, nb_);
    auto V_sub = V_.columns(0, nb_);
    auto H_inv_sub = H_inv_.sub_matrix(0, nb_, 0, nb_);
    auto H_sub = H_.sub_matrix(0, nb_, 0, nb_);
    
    // D = W^T W
    Ibis::transpose_matmul_small_output(W_sub, W_sub, D_sub);
    
    // H = (D + W^T M^-1 V)^-1
    using TeamPolicy = Kokkos::TeamPolicy<Ibis::Vector<Ibis::real>::exec_space>;
    using TeamMember = typename TeamPolicy::member_type;
    const int W_sub_cols = W_sub.n_cols();
    const int n_rows = W_sub.n_rows();
    const int V_sub_cols = V_sub.n_cols();
    TeamPolicy policy(W_sub_cols * V_sub_cols, Kokkos::AUTO);
    Kokkos::parallel_for(
        "matmul_small_output", policy, KOKKOS_LAMBDA(const TeamMember& team) {
        const int idx = team.league_rank();
        const int i = idx / V_sub_cols;
        const int j = idx % V_sub_cols;
 
        Ibis::real acc = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, n_rows),
            [&](const int k, Ibis::real& local_sum) {
                local_sum += W_sub(k, i) * (1.0 / M(k)) * V_sub(k, j);
            },
            acc);

            // Only one thread per team needs to write the result.
            Kokkos::single(Kokkos::PerTeam(team), [&]() { H_inv_sub(i, j) = D_sub(i, j) - acc; });
    });
    // Kokkos::parallel_for(
    //     "RBP::H+=D", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nb_, nb_}),
    //     KOKKOS_LAMBDA(const int i, const int j) {
    //         H_inv_sub(i, j) += D_sub(i, j);  
    //     }
    // );
    Ibis::invert_square_matrix(H_inv_sub, H_sub);
    // std::cout << D_sub(0, 0) << " " << H_inv_sub(0, 0) << " " << H_sub(0, 0) << std::endl;
}

template <typename MemModel>
void ReducedBasisPreconditioner<MemModel>::solve(Ibis::Vector<Ibis::real>& rhs,
                                                 Ibis::Vector<Ibis::real>& x) {
    if (nb_ == 0) {
        x.deep_copy_space(rhs);
        return;
    }

    // We work on the sub-matrices of the number of basis vectors
    // we currently have
    auto W_sub = W_.columns(0, nb_);
    auto V_sub = V_.columns(0, nb_);
    auto H_sub = H_.sub_matrix(0, nb_, 0, nb_);
    auto w_sub = w_.sub_vector(0, nb_);
    auto p_sub = p_.sub_vector(0, nb_);
    auto z = z_;
    auto s = s_;
    auto M = M_;

    Kokkos::parallel_for(
        "RBP::z", n_vars_, KOKKOS_LAMBDA(const int i) {
            z(i) = 1.0 / M(i) * rhs(i);
        }
    );

    Ibis::mat_transpose_vec_small_output(W_sub, z, w_sub); // w = W^T * rhs
    Ibis::gemv(H_sub, w_sub, p_sub); // p = H w
    Ibis::gemv(V_sub, p_sub, d_); // d = V p

    // P^-1 y = rhs - d
    auto d = d_;
    Kokkos::parallel_for("RBP::z", n_vars_, KOKKOS_LAMBDA(const size_t i){
        x(i) = z(i) - 1.0 / M(i) * d(i);              
    });
}

template class ReducedBasisPreconditioner<SharedMem>;
template class ReducedBasisPreconditioner<Mpi>;
