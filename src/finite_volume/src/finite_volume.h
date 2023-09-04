#ifndef FINITE_VOLUME_H
#define FINITE_VOLUME_H

#include "../../grid/src/grid.h"
#include "../../gas/src/flow_state.h"
#include "conserved_quantities.h"


template <typename T>
class FiniteVolume {
public:
    void compute_dudt(const FlowStates<T>& flow_state, 
                      const GridBlock<T>& grid,
                      ConservedQuantities<T>& dudt);

private:
    FlowStates<T> left_;
    FlowStates<T> right_;
    ConservedQuantities<double> flux_;
};

#endif
