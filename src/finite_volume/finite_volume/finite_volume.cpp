#include <finite_volume/finite_volume.h>
#include <finite_volume/boundaries/boundary.h>
#include <finite_volume/conserved_quantities.h>
#include <finite_volume/flux_calc.h>

template <typename T>
FiniteVolume<T>::FiniteVolume(const GridBlock<T>& grid, json config)
    : left_(FlowStates<T>(grid.num_interfaces())),
      right_(FlowStates<T>(grid.num_interfaces())),
      flux_(ConservedQuantities<T>(grid.num_interfaces(), grid.dim())) {
    dim_ = grid.dim();
    json convective_flux_config = config.at("convective_flux");
    reconstruction_order_ = convective_flux_config.at("reconstruction_order");
    flux_calculator_ = flux_calculator_from_string(
        convective_flux_config.at("flux_calculator"));

    std::vector<std::string> boundary_tags = grid.boundary_tags();
    json boundaries_config = config.at("grid").at("boundaries");
    for (unsigned int bi = 0; bi < boundary_tags.size(); bi++) {
        // build the actual boundary condition
        json boundary_config = boundaries_config.at(boundary_tags[bi]);
        std::shared_ptr<BoundaryCondition<T>> boundary(
            new BoundaryCondition<T>(boundary_config));
        bcs_.push_back(boundary);

        // the faces associated with this boundary
        bc_interfaces_.push_back(grid.boundary_faces(boundary_tags[bi]));
    }
}

template <typename T>
int FiniteVolume<T>::compute_dudt(FlowStates<T>& flow_state,
                                  const GridBlock<T>& grid,
                                  ConservedQuantities<T>& dudt,
                                  IdealGas<T>& gas_model) {
    apply_pre_reconstruction_bc(flow_state, grid);
    reconstruct(flow_state, grid, reconstruction_order_);
    compute_flux(grid, gas_model);
    flux_surface_integral(grid, dudt);
    return 0;
}

template <typename T>
double FiniteVolume<T>::estimate_dt(const FlowStates<T>& flow_state,
                                    GridBlock<T>& grid,
                                    IdealGas<T>& gas_model) {
    int num_cells = grid.num_cells();
    CellFaces<T> cell_interfaces = grid.cells().faces();
    Interfaces<T> interfaces = grid.interfaces();
    Cells<T> cells = grid.cells();
    // IdealGas<T> gas_model = gas_model_;

    double dt;
    Kokkos::parallel_reduce(
        "FV::signal_frequency", num_cells,
        KOKKOS_LAMBDA(const int cell_i, double& dt_utd) {
            auto cell_face_ids = cell_interfaces.face_ids(cell_i);
            T spectral_radii = 0.0;
            for (unsigned int face_idx = 0; face_idx < cell_face_ids.size();
                 face_idx++) {
                int i_face = cell_face_ids(face_idx);
                T dot = flow_state.vel.x(cell_i) * interfaces.norm().x(i_face) +
                        flow_state.vel.y(cell_i) * interfaces.norm().y(i_face) +
                        flow_state.vel.z(cell_i) * interfaces.norm().z(i_face);
                T sig_vel = Kokkos::fabs(dot) +
                            gas_model.speed_of_sound(flow_state.gas, cell_i);
                spectral_radii += sig_vel * interfaces.area(i_face);
            }
            T local_dt = cells.volume(cell_i) / spectral_radii;
            dt_utd = Kokkos::min(local_dt, dt_utd);
        },
        Kokkos::Min<double>(dt));

    return dt;
}

template <typename T>
void FiniteVolume<T>::apply_pre_reconstruction_bc(FlowStates<T>& fs,
                                                  const GridBlock<T>& grid) {
    for (unsigned int i = 0; i < bcs_.size(); i++) {
        std::shared_ptr<BoundaryCondition<T>> bc = bcs_[i];
        Field<int> bc_faces = bc_interfaces_[i];
        bc->apply_pre_reconstruction(fs, grid, bc_faces);
    }
}

template <typename T>
void FiniteVolume<T>::reconstruct(FlowStates<T>& flow_states,
                                  const GridBlock<T>& grid,
                                  unsigned int order) {
    (void)order;
    int n_faces = grid.num_interfaces();
    FlowStates<T> this_left = left_;
    FlowStates<T> this_right = right_;
    Interfaces<T> interfaces = grid.interfaces();
    Kokkos::parallel_for(
        "Reconstruct", n_faces, KOKKOS_LAMBDA(const int i_face) {
            // copy left flow states
            int left = interfaces.left_cell(i_face);
            this_left.gas.temp(i_face) = flow_states.gas.temp(left);
            this_left.gas.pressure(i_face) = flow_states.gas.pressure(left);
            this_left.gas.rho(i_face) = flow_states.gas.rho(left);
            this_left.gas.energy(i_face) = flow_states.gas.energy(left);
            this_left.vel.x(i_face) = flow_states.vel.x(left);
            this_left.vel.y(i_face) = flow_states.vel.y(left);
            this_left.vel.z(i_face) = flow_states.vel.z(left);

            // copy right flow states
            int right = interfaces.right_cell(i_face);
            this_right.gas.temp(i_face) = flow_states.gas.temp(right);
            this_right.gas.pressure(i_face) = flow_states.gas.pressure(right);
            this_right.gas.rho(i_face) = flow_states.gas.rho(right);
            this_right.gas.energy(i_face) = flow_states.gas.energy(right);
            this_right.vel.x(i_face) = flow_states.vel.x(right);
            this_right.vel.y(i_face) = flow_states.vel.y(right);
            this_right.vel.z(i_face) = flow_states.vel.z(right);
        });
}

template <typename T>
void FiniteVolume<T>::compute_flux(const GridBlock<T>& grid,
                                   IdealGas<T>& gas_model) {
    // rotate velocities to the interface local frames
    Interfaces<T> faces = grid.interfaces();
    transform_to_local_frame(left_.vel, faces.norm(), faces.tan1(),
                             faces.tan2());
    transform_to_local_frame(right_.vel, faces.norm(), faces.tan1(),
                             faces.tan2());

    switch (flux_calculator_) {
        case FluxCalculator::Hanel:
            hanel(left_, right_, flux_, gas_model, dim_ == 3);
            break;
        case FluxCalculator::Ausmdv:
            ausmdv(left_, right_, flux_, gas_model, dim_ == 3);
            break;
    }

    // rotate the fluxes to the global frame
    Vector3s<T> norm = faces.norm();
    Vector3s<T> tan1 = faces.tan1();
    Vector3s<T> tan2 = faces.tan2();
    ConservedQuantities<T> flux = flux_;
    Kokkos::parallel_for(
        "flux::transform_to_global", faces.size(), KOKKOS_LAMBDA(const int i) {
            T px = flux.momentum_x(i);
            T py = flux.momentum_y(i);
            T pz = 0.0;
            if (flux.dim() == 3) {
                pz = flux.momentum_z(i);
            }
            T x = px * norm.x(i) + py * tan1.x(i) + pz * tan2.x(i);
            T y = px * norm.y(i) + py * tan1.y(i) + pz * tan2.y(i);
            T z = px * norm.z(i) + py * tan1.z(i) + pz * tan2.z(i);
            flux.momentum_x(i) = x;
            flux.momentum_y(i) = y;
            if (flux.dim() == 3) {
                flux.momentum_z(i) = z;
            }
        });
}

template <typename T>
void FiniteVolume<T>::flux_surface_integral(const GridBlock<T>& grid,
                                            ConservedQuantities<T>& dudt) {
    Cells<T> cells = grid.cells();
    CellFaces<T> cell_faces = grid.cells().faces();
    Interfaces<T> faces = grid.interfaces();
    ConservedQuantities<T> flux = flux_;
    int num_cells = grid.num_cells();
    Kokkos::parallel_for(
        "flux_integral", num_cells, KOKKOS_LAMBDA(const int cell_i) {
            auto face_ids = cell_faces.face_ids(cell_i);
            T d_mass = 0.0;
            T d_momentum_x = 0.0;
            T d_momentum_y = 0.0;
            T d_momentum_z = 0.0;
            T d_energy = 0.0;
            for (unsigned int face_i = 0; face_i < face_ids.size(); face_i++) {
                int face_id = face_ids(face_i);
                T area =
                    -faces.area(face_id) * cell_faces.outsigns(cell_i)(face_i);
                d_mass += flux.mass(face_id) * area;
                d_momentum_x += flux.momentum_x(face_id) * area;
                d_momentum_y += flux.momentum_y(face_id) * area;
                if (flux.dim() == 3) {
                    d_momentum_z += flux.momentum_z(face_id) * area;
                }
                d_energy += flux.energy(face_id) * area;
            }
            dudt.mass(cell_i) = d_mass / cells.volume(cell_i);
            dudt.momentum_x(cell_i) = d_momentum_x / cells.volume(cell_i);
            dudt.momentum_y(cell_i) = d_momentum_y / cells.volume(cell_i);
            if (dudt.dim() == 3) {
                dudt.momentum_z(cell_i) = d_momentum_z / cells.volume(cell_i);
            }
            dudt.energy(cell_i) = d_energy / cells.volume(cell_i);
        });
}

template <typename T>
int FiniteVolume<T>::count_bad_cells(const FlowStates<T>& fs,
                                     const int num_cells) {
    int n_bad_cells = 0;
    Kokkos::parallel_reduce(
        "FiniteVolume::count_bad_cells", num_cells,
        KOKKOS_LAMBDA(const int cell_i, int& n_bad_cells_utd) {
            if (fs.gas.temp(cell_i) < 0.0 || fs.gas.rho(cell_i) < 0.0) {
                n_bad_cells_utd += 1;
            }
        },
        n_bad_cells);
    return n_bad_cells;
}

template class FiniteVolume<double>;
