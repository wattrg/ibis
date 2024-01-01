#include <finite_volume/boundaries/boundary.h>
#include <finite_volume/conserved_quantities.h>
#include <finite_volume/finite_volume.h>
#include <finite_volume/flux_calc.h>

#include <stdexcept>

#include "finite_volume/gradient.h"

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

    if (reconstruction_order_ == 2) {
        grad_calc_ = WLSGradient<T>(grid);
        grad_p_ = Vector3s<T>("FV::grad_p", grid.num_cells());
        grad_rho_ = Vector3s<T>("FV::grad_rho", grid.num_cells());
        grad_vx_ = Vector3s<T>("FV::grad_vx", grid.num_cells());
        grad_vy_ = Vector3s<T>("FV::grad_vy", grid.num_cells());
        grad_vz_ = Vector3s<T>("FV::grad_vz", grid.num_cells());
    }

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
    reconstruct(flow_state, grid, gas_model, reconstruction_order_);
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
                                  IdealGas<T>& gas_model,
                                  unsigned int order) {
    (void)order;
    switch (order) {
        case 1:
            copy_reconstruct(flow_states, grid);
            break;
        case 2:
            linear_reconstruct(flow_states, grid, gas_model);
            break;
        default:
            spdlog::error("Invalid reconstruction order {}", order);
            throw std::runtime_error("Invalid reconstruction order");
    }
}

template <typename T>
void FiniteVolume<T>::copy_reconstruct(FlowStates<T>& flow_states,
                                       const GridBlock<T>& grid) {
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
KOKKOS_INLINE_FUNCTION T linear_interpolate(T value, Vector3s<T> grad, T dx,
                                            T dy, T dz, int i, bool is_valid) {
    T grad_x = 0.0;
    T grad_y = 0.0;
    T grad_z = 0.0;
    if (is_valid) {
        grad_x = grad.x(i);
        grad_y = grad.y(i);
        grad_z = grad.z(i);
    }


    return value + grad_x * dx + grad_y * dy + grad_z * dz;
}

template <typename T>
void FiniteVolume<T>::linear_reconstruct(FlowStates<T>& flow_states,
                                         const GridBlock<T>& grid,
                                         IdealGas<T>& gas_model) {
    grad_calc_.compute_gradients(grid, flow_states.gas.pressure(), grad_p_);
    grad_calc_.compute_gradients(grid, flow_states.gas.rho(), grad_rho_);
    grad_calc_.compute_gradients(grid, flow_states.vel.x(), grad_vx_);
    grad_calc_.compute_gradients(grid, flow_states.vel.y(), grad_vy_);
    grad_calc_.compute_gradients(grid, flow_states.vel.z(), grad_vz_);

    auto cells = grid.cells();
    auto faces = grid.interfaces();
    auto left = left_;
    auto right = right_;
    auto grad_p = grad_p_;
    auto grad_rho = grad_rho_;
    auto grad_vx = grad_vx_;
    auto grad_vy = grad_vy_;
    auto grad_vz = grad_vz_;
    int num_cells = grid.num_cells();
    Kokkos::parallel_for(
        "FV::linear_reconstruct", grid.num_interfaces(),
        KOKKOS_LAMBDA(const int i_face) {
            int left_cell = faces.left_cell(i_face);
            bool left_valid = left_cell < num_cells;
            T dx = faces.centre().x(i_face) - cells.centroids().x(left_cell);
            T dy = faces.centre().y(i_face) - cells.centroids().y(left_cell);
            T dz = faces.centre().z(i_face) - cells.centroids().z(left_cell);
            left.gas.pressure(i_face) =
                linear_interpolate(flow_states.gas.pressure(left_cell), grad_p,
                                   dx, dy, dz, left_cell, left_valid);
            left.gas.rho(i_face) =
                linear_interpolate(flow_states.gas.rho(left_cell), grad_rho, dx,
                                   dy, dz, left_cell, left_valid);
            left.vel.x(i_face) = linear_interpolate(
                flow_states.vel.x(left_cell), grad_vx, dx, dy, dz, left_cell, left_valid);
            left.vel.y(i_face) = linear_interpolate(
                flow_states.vel.y(left_cell), grad_vy, dx, dy, dz, left_cell, left_valid);
            left.vel.z(i_face) = linear_interpolate(
                flow_states.vel.z(left_cell), grad_vz, dx, dy, dz, left_cell, left_valid);
            gas_model.update_thermo_from_rhop(left.gas, i_face);

            int right_cell = faces.right_cell(i_face);
            bool right_valid = right_cell < num_cells;
            dx = faces.centre().x(i_face) - cells.centroids().x(right_cell);
            dy = faces.centre().y(i_face) - cells.centroids().y(right_cell);
            dz = faces.centre().z(i_face) - cells.centroids().z(right_cell);
            right.gas.pressure(i_face) =
                linear_interpolate(flow_states.gas.pressure(right_cell), grad_p,
                                   dx, dy, dz, right_cell, right_valid);
            right.gas.rho(i_face) =
                linear_interpolate(flow_states.gas.rho(right_cell), grad_rho,
                                   dx, dy, dz, right_cell, right_valid);
            right.vel.x(i_face) = linear_interpolate(
                flow_states.vel.x(right_cell), grad_vx, dx, dy, dz, right_cell, right_valid);
            right.vel.y(i_face) = linear_interpolate(
                flow_states.vel.y(right_cell), grad_vy, dx, dy, dz, right_cell, right_valid);
            right.vel.z(i_face) = linear_interpolate(
                flow_states.vel.z(right_cell), grad_vz, dx, dy, dz, right_cell, right_valid);
            gas_model.update_thermo_from_rhop(right.gas, i_face);
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
            if (fs.gas.temp(cell_i) < 0.0 || fs.gas.rho(cell_i) < 0.0 ||
                Kokkos::isnan(fs.gas.rho(cell_i)) ||
                Kokkos::isinf(fs.gas.rho(cell_i))) {
                n_bad_cells_utd += 1;
            }
        },
        n_bad_cells);
    return n_bad_cells;
}

template class FiniteVolume<double>;
