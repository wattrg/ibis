#ifndef FINITE_VOLUME_H
#define FINITE_VOLUME_H

#include "../../grid/src/grid.h"
#include "../../gas/src/flow_state.h"
#include "conserved_quantities.h"


template <typename T>
class FiniteVolume {
public:
    FiniteVolume();

    FiniteVolume(unsigned int n_cells, unsigned int n_interfaces, unsigned int dim)
        : left_(FlowStates<T>("Left", n_interfaces)),
          right_(FlowStates<T>("Right", n_interfaces)),
          flux_(ConservedQuantities<T>(n_interfaces, dim)) {}

    int compute_dudt(const FlowStates<T>& flow_state, 
                      const GridBlock<T>& grid,
                      ConservedQuantities<T>& dudt);


private:
    // memory
    FlowStates<T> left_;
    FlowStates<T> right_;
    ConservedQuantities<double> flux_;

    // configuration
    unsigned int dim;
    unsigned int reconstruction_order;

    // methods
    apply_pre_reconstruction_bc();
    reconstruct(FlowStates<T>& flow_states, unsigned int reconstruction_order){}
    compute_flux();
    apply_post_convective_flux_bc(){}
    apply_pre_spatial_deriv(){}

};

#endif
