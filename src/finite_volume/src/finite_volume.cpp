#include "finite_volume.h"
#include "boundaries/boundary.h"
#include "conserved_quantities.h"
#include "flux_calc.h"

template <typename T>
FiniteVolume<T>::FiniteVolume(const GridBlock<T>& grid, json config)
    : left_(FlowStates<T>(grid.num_interfaces())),
      right_(FlowStates<T>(grid.num_interfaces())),
      flux_(ConservedQuantities<T>(grid.num_interfaces(), grid.dim()))
{
    dim_ = grid.dim();
    json convective_flux_config = config.at("convective_flux");
    reconstruction_order_ = convective_flux_config.at("reconstruction_order");
    flux_calculator_ = flux_calculator_from_string(
        convective_flux_config.at("flux_calculator")
    );

    std::vector<std::string> boundary_tags = grid.boundary_tags(); 
    json boundaries_config = config.at("grid").at("boundaries");
    for (unsigned int bi = 0; bi < boundary_tags.size(); bi++){
        // build the actual boundary condition
        json boundary_config = boundaries_config.at(boundary_tags[bi]);
        std::shared_ptr<BoundaryCondition<T>> boundary(new BoundaryCondition<T>(boundary_config));
        bcs_.push_back(boundary);

        // the faces associated with this boundary
        bc_interfaces_.push_back(grid.boundary_faces(boundary_tags[bi]));
    }
}

template <typename T>
int FiniteVolume<T>::compute_dudt(FlowStates<T>& flow_state,
                                  const GridBlock<T>& grid,
                                  ConservedQuantities<T>& dudt)
{
    apply_pre_reconstruction_bc(flow_state, grid);
    reconstruct(flow_state, grid, reconstruction_order_);
    compute_flux(grid);
    flux_surface_integral(grid, dudt);
    return 0;
}

template <typename T>
double FiniteVolume<T>::estimate_signal_frequency(const FlowStates<T> &flow_state, GridBlock<T> &grid) {
    int num_cells = grid.num_cells();
    CellFaces<T> cell_interfaces_ids = grid.cells().faces();
    Interfaces<T> interfaces = grid.interfaces();
    Cells<T> cells = grid.cells();
    double signal_frequency = 0.0;

    Kokkos::parallel_reduce("FV::signal_frequency", 
                            num_cells, 
                            KOKKOS_LAMBDA(const int cell_i, double& signal_frequency_utd) 
    {
        auto cell_face_ids = cell_interfaces_ids.face_ids(cell_i);
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

template <typename T>
void FiniteVolume<T>::apply_pre_reconstruction_bc(FlowStates<T>& fs, const GridBlock<T>& grid) {
    for (unsigned int i = 0; i < bcs_.size(); i++) {
        std::shared_ptr<BoundaryCondition<T>> bc = bcs_[i];
        Field<int> bc_faces = bc_interfaces_[i];
        bc->apply_pre_reconstruction(fs, grid, bc_faces);
    } 
}

template <typename T>
void FiniteVolume<T>::reconstruct(FlowStates<T>& flow_states, const GridBlock<T>& grid, unsigned int order){
    (void) order;
    int n_faces = grid.num_interfaces();
    Kokkos::parallel_for("Reconstruct", n_faces, KOKKOS_LAMBDA(const int i_face){
        // copy left flow states
        int left = grid.interfaces().left_cell(i_face);
        left_.gas.temp(i_face) = flow_states.gas.temp(left);
        left_.gas.pressure(i_face) = flow_states.gas.pressure(left);
        left_.gas.rho(i_face) = flow_states.gas.rho(left);
        left_.gas.energy(i_face) = flow_states.gas.energy(left);

        // copy right flow states
        int right = grid.interfaces().right_cell(i_face);
        right_.gas.temp(i_face) = flow_states.gas.temp(right);
        right_.gas.pressure(i_face) = flow_states.gas.pressure(right);
        right_.gas.rho(i_face) = flow_states.gas.rho(right);
        right_.gas.energy(i_face) = flow_states.gas.energy(right);
    });
}

template <typename T>
void FiniteVolume<T>::compute_flux(const GridBlock<T>& grid) {
    Interfaces<T> faces = grid.interfaces();
    transform_to_local_frame(left_.vel, faces.norm(), faces.tan1(), faces.tan2());
    transform_to_local_frame(right_.vel, faces.norm(), faces.tan1(), faces.tan2());

    switch (flux_calculator_) {
        case FluxCalculator::Hanel:
            hanel(left_, right_, flux_, dim_==3);
            break;
    }

    // do we need to transform the flow states back to the global reference frame?
}

template <typename T>
void FiniteVolume<T>::flux_surface_integral(const GridBlock<T>& grid, ConservedQuantities<T>& dudt){
    Cells<T> cells = grid.cells();
    CellFaces<T> cell_faces = grid.cells().faces();
    Interfaces<T> faces = grid.interfaces();
    Kokkos::parallel_for("flux_integral", grid.num_cells(), KOKKOS_LAMBDA(const int cell_i){
        auto face_ids = cell_faces.face_ids(cell_i);
        T d_mass = 0.0;
        T d_momentum_x = 0.0;
        T d_momentum_y = 0.0;
        T d_momentum_z = 0.0;
        T d_energy = 0.0;
        for (unsigned int face_i = 0; face_i < face_ids.size(); face_i++){
            int face_id = face_ids(face_i); 
            T area = faces.area(face_id)* cell_faces.outsigns(cell_i)(face_i);
            d_mass += flux_.mass(face_id) * area; 
            printf("face_id = %i, mass flux = %f\n", face_id, flux_.mass(face_id));
            d_momentum_x += flux_.momentum_x(face_id) * area; 
            d_momentum_y += flux_.momentum_y(face_id) * area; 
            d_momentum_z += flux_.momentum_z(face_id) * area; 
            d_energy += flux_.energy(face_id) * area;
        }
        dudt.mass(cell_i) = d_mass / cells.volume(cell_i);
        dudt.momentum_x(cell_i) = d_momentum_x / cells.volume(cell_i);
        dudt.momentum_y(cell_i) = d_momentum_y / cells.volume(cell_i);
        dudt.momentum_z(cell_i) = d_momentum_z / cells.volume(cell_i);
        dudt.energy(cell_i) = d_energy / cells.volume(cell_i);
    });
}

template class FiniteVolume<double>;
