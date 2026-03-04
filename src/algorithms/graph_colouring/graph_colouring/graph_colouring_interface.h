#ifndef GRAPH_COLOURING_INTERFACE_H
#define GRAPH_COLOURING_INTERFACE_H

#include <util/types.h>

template <class GridBlock_type>
class GridColourer {
public:
    virtual ~GridColourer() = default;

    virtual void compute_colouring(const GridBlock_type& grid) = 0;

    virtual Ibis::Array1D<int> colours() const = 0;

    virtual int num_colours() const = 0;
};

#endif
