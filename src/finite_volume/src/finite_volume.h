#ifndef FINITE_VOLUME_H
#define FINITE_VOLUME_H

#include "../../grid/src/grid.h"
#include "../../gas/src/flow_state.h"
#include "conserved_quantities.h"
#include "boundaries/boundary.h"
#include "impl/Kokkos_HostThreadTeam.hpp"


template <typename T>
class FiniteVolume {
public:
    FiniteVolume();

    FiniteVolume(const GridBlock<T>& grid);

    int compute_dudt(const FlowStates<T>& flow_state, 
                      const GridBlock<T>& grid,
                      ConservedQuantities<T>& dudt);

    double estimate_signal_frequency(const FlowStates<T>& flow_state, GridBlock<T>& grid);


private:
    // memory
    FlowStates<T> left_;
    FlowStates<T> right_;
    ConservedQuantities<double> flux_;

    // ghost cells
    // FlowStates<T> ghost_;

    // boundary conditions
    // NOTE: these will become vectors of these properties. 
    // But for the moment, I'm just using a single boundary 
    // condition to make sure everything else works
    // SupersonicInflow<T> bc_;
    // Field<int> bc_interfaces_;

    // configuration
    unsigned int dim;
    unsigned int reconstruction_order;

    // methods
    void apply_pre_reconstruction_bc();
    void reconstruct(FlowStates<T>& flow_states, unsigned int order);
    void compute_flux();
    void apply_post_convective_flux_bc();
    void apply_pre_spatial_deriv();

};

#endif
