#include <linear_algebra/ilu.h>

template <class MemModel>
ILU<MemModel>::ILU(std::shared_ptr<LinearSystem> system, size_t k) : system_(system) {
    // construct the graph of the sparsity pattern of the
    // linear system
    Ibis::CrsGraph<int, int> graph = system->compute_matrix_graph();

    // construct the ilu implementation handle
    ilu_handle_ = ILU::ilu_handle_type(graph, k);

    // construct the triangular solver implementation handles
    lower_triangular_solve_handle_ = ILU::triangular_solver_handle_type(
        ilu_handle_.L, Ibis::TriangularMatrixType::LOWER);
    lower_triangular_solve_handle_ = ILU::triangular_solver_handle_type(
        ilu_handle_.U, Ibis::TriangularMatrixType::UPPER);

    temp_vec_ = Ibis::Vector<Ibis::real>("ILU::temp_vec", system->num_vars());
}

template <class MemModel>
void ILU<MemModel>::decompose() {
    system_->compute_matrix(matrix_);
    ilu_handle_.numeric_phase(matrix_);
}

template <class MemModel>
void ILU<MemModel>::solve(Ibis::Vector<Ibis::real>& x) {
    lower_triangular_solve_handle_.solve(ilu_handle_.L, system_->rhs(), temp_vec_);
    upper_triangular_solve_handle_.solve(ilu_handle_.U, temp_vec_, x);
}
