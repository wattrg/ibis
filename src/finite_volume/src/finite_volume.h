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

    FiniteVolume(const GridBlock<T>& grid)
        : left_(FlowStates<T>(grid.num_interfaces())),
          right_(FlowStates<T>(grid.num_interfaces())),
          flux_(ConservedQuantities<T>(grid.num_interfaces(), grid.dim()))
    {}

    int compute_dudt(const FlowStates<T>& flow_state, 
                      const GridBlock<T>& grid,
                      ConservedQuantities<T>& dudt)
    {
        (void) flow_state;
        (void) grid;
        (void) dudt;
        return 0;
    }

    double estimate_signal_frequency(const FlowStates<T>& flow_state, GridBlock<T>& grid){
        int num_cells = grid.num_cells();
        Id cell_interfaces_ids = grid.cells().interface_ids();
        Interfaces<T> interfaces = grid.interfaces();
        Cells<T> cells = grid.cells();
        double signal_frequency = 0.0;

        Kokkos::parallel_reduce("FV::signal_frequency", 
                                num_cells, 
                                KOKKOS_LAMBDA(const int cell_i, double& signal_frequency_utd) 
        {
            auto cell_face_ids = cell_interfaces_ids[cell_i];
            T spectral_radii = 0.0;
            for (unsigned int face_idx = 0; face_idx < cell_face_ids.size(); face_idx++) {
                int i_face = cell_face_ids(face_idx);
                T x_term = flow_state.vel.x(cell_i) * interfaces.norm().x(i_face);
                T y_term = flow_state.vel.y(cell_i) * interfaces.norm().y(i_face);
                T z_term = flow_state.vel.z(cell_i) * interfaces.norm().z(i_face);
                T dot = x_term + y_term + z_term;
                T sig_vel = Kokkos::fabs(dot) + Kokkos::sqrt(1.4*287*flow_state.gas.temp(cell_i));
                spectral_radii += sig_vel * interfaces.area(i_face); 
            }
            T local_signal_frequency = spectral_radii / cells.volume(cell_i);
            signal_frequency_utd = Kokkos::max(local_signal_frequency, signal_frequency_utd);
        }, signal_frequency);

        return signal_frequency;
    }


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
