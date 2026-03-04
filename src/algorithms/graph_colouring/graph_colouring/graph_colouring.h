#ifndef COLOURING_H
#define COLOURING_H

#include <graph_colouring/graph_colouring_interface.h>
#include <graph_colouring_kokkos_kernels/graph_colouring_kokkos_kernels.h>

template <typename GridBlock_type>
std::unique_ptr<GridColourer<GridBlock_type>> make_grid_colourer() {
    return std::make_unique<KokkosKernels_GridColourer<GridBlock_type>>();
}

#endif
