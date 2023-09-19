#include "finite_volume.h"
#include "conserved_quantities.h"

template <typename T>
FiniteVolume<T>::FiniteVolume(const GridBlock<T>& grid)
    : left_(FlowStates<T>(grid.num_interfaces())),
      right_(FlowStates<T>(grid.num_interfaces())),
      flux_(ConservedQuantities<T>(grid.num_interfaces(), grid.dim()))
{}

template <typename T>
int FiniteVolume<T>::compute_dudt(const FlowStates<T>& flow_state,
                                  const GridBlock<T>& grid,
                                  ConservedQuantities<T>& dudt)
{
    (void) flow_state;
    (void) grid;
    (void) dudt;
    return 0;
}

template <typename T>
double FiniteVolume<T>::estimate_signal_frequency(const FlowStates<T> &flow_state, GridBlock<T> &grid) {
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

template class FiniteVolume<double>;

// template <typename T>
// int FiniteVolume<double>::compute_dudt(const FlowStates<T>& flow_state,
//                                const GridBlock<T>& grid,
//                                ConservedQuantities<T>& dudt)
// {
//     apply_prep_reconstruction_bc();
//     reconstruct(flow_state); 
// }
