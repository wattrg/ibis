#ifndef COLOURING_H
#define COLOURING_H

#include <util/types.h>

template <class GridBlock_type>
class GridColourer {
public:
    virtual void compute_colouring(const GridBlock_type& grid) = 0;

    virtual Ibis::Array1D<int> colours() const = 0;

    virtual int num_colours() const = 0;
};

template <typename GridBlock_type>
class KokkosKernels_GridColourer;

template <typename GridBlock_type>
std::unique_ptr<GridColourer<GridBlock_type>> make_grid_colourer() {
    return std::make_unique<GridColourer<GridBlock_type>>(
        KokkosKernels_GridColourer<GridBlock_type>());
}

#endif
