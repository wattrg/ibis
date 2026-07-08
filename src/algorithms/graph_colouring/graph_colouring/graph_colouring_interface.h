#ifndef GRAPH_COLOURING_INTERFACE_H
#define GRAPH_COLOURING_INTERFACE_H

#include <util/types.h>

template <class CrsGraphType>
class GraphColourer {
public:
    virtual ~GraphColourer() = default;

    virtual void compute_colouring(const CrsGraphType& graph) = 0;

    virtual Ibis::Array1D<int> colours() const = 0;

    virtual int num_colours() const = 0;
};

#endif
