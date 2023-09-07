#ifndef FINITE_VOLUME_H
#define FINITE_VOLUME_H

#include "../../grid/src/grid.h"
#include "../../gas/src/flow_state.h"
#include "conserved_quantities.h"
#include "boundaries/boundary.h"


template <typename T>
class FiniteVolume {
public:
    FiniteVolume();

    FiniteVolume(const GridBlock<T>& grid)
        : left_(FlowStates<T>("Left", grid.num_interfaces())),
          right_(FlowStates<T>("Right", grid.num_interfaces())),
          flux_(ConservedQuantities<T>(grid.num_interfaces(), grid.dim()))
    {}

    int compute_dudt(const FlowStates<T>& flow_state, 
                      const GridBlock<T>& grid,
                      ConservedQuantities<T>& dudt);


private:
    // memory
    FlowStates<T> left_;
    FlowStates<T> right_;
    ConservedQuantities<double> flux_;

    // ghost cells
    FlowStates<T> ghost_;

    // boundary condition
    SupersonicInflow<T> bc_;

    // configuration
    unsigned int dim;
    unsigned int reconstruction_order;

    // methods
    void apply_pre_reconstruction_bc();
    void reconstruct(FlowStates<T>& flow_states, unsigned int order){}
    void compute_flux();
    void apply_post_convective_flux_bc(){}
    void apply_pre_spatial_deriv(){}

};

#endif
