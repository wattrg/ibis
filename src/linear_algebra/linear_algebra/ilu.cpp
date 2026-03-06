#include <linear_algebra/ilu.h>

template <class MemModel>
ILU<MemModel>::ILU(std::shared_ptr<LinearSystem> system, size_t k, int stencil_distance)
    : system_(system) {
    // construct the graph of the sparsity pattern of the
    // linear system
    Ibis::CrsGraph<int, int> graph = system->compute_matrix_graph(stencil_distance);
    Ibis::Array1D<Ibis::real> values("ILU::matrix::values", graph.num_entries());
    matrix_ = Ibis::CrsMatrix<int, int, Ibis::real>(graph, values);

    // construct the ilu implementation handle
    ilu_handle_ = std::make_unique<ILU::ilu_handle_type>(graph, k);

    // construct the triangular solver implementation handles
    lower_triangular_solve_handle_ = std::make_unique<ILU::triangular_solver_handle_type>(
        ilu_handle_->L, Ibis::TriangularMatrixType::LOWER);
    upper_triangular_solve_handle_ = std::make_unique<ILU::triangular_solver_handle_type>(
        ilu_handle_->U, Ibis::TriangularMatrixType::UPPER);

    // Compute and decompose the matrix
    // update_preconditioner();

    temp_vec_ = Ibis::Vector<Ibis::real>("ILU::temp_vec", system->num_vars());
}

template <class MemModel>
void ILU<MemModel>::update_preconditioner() {
    // evaluate the matrix again
    system_->compute_matrix(matrix_);

    // decompose the matrix
    ilu_handle_->numeric_phase(matrix_);
}

template <class MemModel>
void ILU<MemModel>::solve(Ibis::Vector<Ibis::real>& rhs, Ibis::Vector<Ibis::real>& x) {
    lower_triangular_solve_handle_->solve(ilu_handle_->L, rhs, temp_vec_);
    upper_triangular_solve_handle_->solve(ilu_handle_->U, temp_vec_, x);
}

template class ILU<SharedMem>;
