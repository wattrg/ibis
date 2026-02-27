#ifndef COLOURING_H
#define COLOURING_H

#include <graph_colouring_kokkos_kernels/graph_colouring_kokkos_kernels.h>
#include <util/types.h>

template <class GridBlock_type>
class GridColourer {
public:
    virtual void compute_colouring(const GridBlock_type& grid) = 0;

    virtual Ibis::Array1D<int> colours() const = 0;

    virtual size_t num_colours() const = 0;
};

template <class GridBlock_type>
std::unique_ptr<GridColourer<GridBlock_type>> make_grid_colourer() {
    return std::make_unique<GridBlock_type>(
        KokkosKernels_GridColourer<GridBlock_type>()
    );
}

#endif
