#include <linear_algebra/ilu.h>

template <class MemModel, typename T>
ILU<MemModel, T>::ILU(GridBlock<MemModel, T> grid, std::shared_ptr<LinearSystem> system,
                      size_t k)
    : system_(system) {
    // Step 1: construct the graph
    grid.compute_graph(2);
    auto graph = grid.graph(2);

    // Step 2: construct the ilu handle
    ilu_handle_ = ILU::handle_type(graph, k);
}

template <class MemModel, typename T>
void ILU<MemModel, T>::decompose() {
    system_->compute_matrix(matrix_);
    ilu_handle_.numeric_phase(matrix_);
}

template <class MemModel, typename T>
LinearSolveResult solver(Ibis::Vector<Ibis::real>& x) {
    // Do the actual solve
}
