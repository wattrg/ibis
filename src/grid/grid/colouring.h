#ifndef COLOURING_H
#define COLOURING_H

#include <grid/grid.h>

template<class GridBlock_type>
class GridColourer {
public:
    virtual Ibis::Array1D<int> colour(const GridBlock_type& grid) = 0;
};

#endif
