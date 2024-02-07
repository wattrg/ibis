#include <finite_volume/boundaries/boundary.h>
#include <finite_volume/conserved_quantities.h>
#include <finite_volume/finite_volume.h>
#include <finite_volume/flux_calc.h>

#include <stdexcept>

#include "gas/transport_properties.h"

template <typename T>
FiniteVolume<T>::FiniteVolume(const GridBlock<T>& grid, json config)
    : left_(FlowStates<T>(grid.num_interfaces())),
      right_(FlowStates<T>(grid.num_interfaces())),
      flux_(ConservedQuantities<T>(grid.num_interfaces(), grid.dim())) {
    dim_ = grid.dim();

    json convective_flux_config = config.at("convective_flux");

    reconstruction_order_ = convective_flux_config.at("reconstruction_order");
    if (reconstruction_order_ > 1) {
        grad_calc_ = WLSGradient<T>(grid);
        grad_ = Gradients<T>(grid.num_cells(), viscous_);
        limiter_ = Limiter<T>(convective_flux_config);
        if (limiter_.enabled()) {
            limiters_ = LimiterValues<T>(grid.num_cells());
        }
    }

    flux_calculator_ = flux_calculator_from_string(
        convective_flux_config.at("flux_calculator"));

    json viscous_flux_config = config.at("viscous_flux");
    viscous_ = viscous_flux_config.at("enabled");

    if (viscous_) {
        face_gs_ = GasStates<T>(grid.num_interfaces());
    }

    std::vector<std::string> boundary_tags = grid.boundary_tags();
    json boundaries_config = config.at("grid").at("boundaries");
    for (size_t bi = 0; bi < boundary_tags.size(); bi++) {
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
size_t FiniteVolume<T>::compute_dudt(FlowStates<T>& flow_state,
                                     const GridBlock<T>& grid,
                                     ConservedQuantities<T>& dudt,
                                     IdealGas<T>& gas_model,
                                     TransportProperties<T>& trans_prop) {
    apply_pre_reconstruction_bc(flow_state, grid);
    reconstruct(flow_state, grid, gas_model, trans_prop, reconstruction_order_);
    compute_convective_flux(grid, gas_model);
    if (viscous_) {
        compute_viscous_flux(flow_state, grid, gas_model, trans_prop);
    }
    flux_surface_integral(grid, dudt);
    return 0;
}

template <typename T>
double FiniteVolume<T>::estimate_dt(const FlowStates<T>& flow_state,
                                    GridBlock<T>& grid, IdealGas<T>& gas_model,
                                    TransportProperties<T>& trans_prop) {
    (void)trans_prop;
    size_t num_cells = grid.num_cells();
    CellFaces<T> cell_interfaces = grid.cells().faces();
    Interfaces<T> interfaces = grid.interfaces();
    Cells<T> cells = grid.cells();
    // IdealGas<T> gas_model = gas_model_;

    double dt;
    Kokkos::parallel_reduce(
        "FV::signal_frequency", num_cells,
        KOKKOS_LAMBDA(const size_t cell_i, double& dt_utd) {
            auto cell_face_ids = cell_interfaces.face_ids(cell_i);
            T spectral_radii = 0.0;
            for (size_t face_idx = 0; face_idx < cell_face_ids.size();
                 face_idx++) {
                size_t i_face = cell_face_ids(face_idx);
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
    for (size_t i = 0; i < bcs_.size(); i++) {
        std::shared_ptr<BoundaryCondition<T>> bc = bcs_[i];
        Field<size_t> bc_faces = bc_interfaces_[i];
        bc->apply_pre_reconstruction(fs, grid, bc_faces);
    }
}

template <typename T>
void FiniteVolume<T>::reconstruct(FlowStates<T>& flow_states,
                                  const GridBlock<T>& grid,
                                  IdealGas<T>& gas_model,
                                  TransportProperties<T>& trans_prop,
                                  size_t order) {
    switch (order) {
        case 1:
            copy_reconstruct(flow_states, grid);
            break;
        case 2:
            linear_reconstruct(flow_states, grid, gas_model, trans_prop);
            break;
        default:
            spdlog::error("Invalid reconstruction order {}", order);
            throw std::runtime_error("Invalid reconstruction order");
    }
}

template <typename T>
void FiniteVolume<T>::copy_reconstruct(FlowStates<T>& flow_states,
                                       const GridBlock<T>& grid) {
    size_t n_faces = grid.num_interfaces();
    FlowStates<T> this_left = left_;
    FlowStates<T> this_right = right_;
    Interfaces<T> interfaces = grid.interfaces();
    Kokkos::parallel_for(
        "Reconstruct", n_faces, KOKKOS_LAMBDA(const size_t i_face) {
            // copy left flow states
            size_t left = interfaces.left_cell(i_face);
            this_left.gas.temp(i_face) = flow_states.gas.temp(left);
            this_left.gas.pressure(i_face) = flow_states.gas.pressure(left);
            this_left.gas.rho(i_face) = flow_states.gas.rho(left);
            this_left.gas.energy(i_face) = flow_states.gas.energy(left);
            this_left.vel.x(i_face) = flow_states.vel.x(left);
            this_left.vel.y(i_face) = flow_states.vel.y(left);
            this_left.vel.z(i_face) = flow_states.vel.z(left);

            // copy right flow states
            size_t right = interfaces.right_cell(i_face);
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
                                            T dy, T dz, int i, T limiter,
                                            bool is_valid) {
    T grad_x = 0.0;
    T grad_y = 0.0;
    T grad_z = 0.0;
    if (is_valid) {
        grad_x = grad.x(i);
        grad_y = grad.y(i);
        grad_z = grad.z(i);
    }

    return value + limiter * (grad_x * dx + grad_y * dy + grad_z * dz);
}

template <typename T>
void FiniteVolume<T>::linear_reconstruct(const FlowStates<T>& flow_states,
                                         const GridBlock<T>& grid,
                                         IdealGas<T>& gas_model,
                                         TransportProperties<T>& trans_prop) {
    (void)trans_prop;
    grad_calc_.compute_gradients(grid, flow_states.gas.pressure(), grad_.p);
    grad_calc_.compute_gradients(grid, flow_states.gas.rho(), grad_.rho);
    grad_calc_.compute_gradients(grid, flow_states.vel.x(), grad_.vx);
    grad_calc_.compute_gradients(grid, flow_states.vel.y(), grad_.vy);
    grad_calc_.compute_gradients(grid, flow_states.vel.z(), grad_.vz);

    auto cells = grid.cells();
    auto faces = grid.interfaces();
    auto left = left_;
    auto right = right_;
    auto grad = grad_;
    auto limiter = limiter_;
    auto limiters = limiters_;
    int num_cells = grid.num_cells();

    bool limiter_enabled = limiter.enabled();
    if (limiter_enabled) {
        limiter.calculate_limiters(flow_states.gas.pressure(), limiters_.p,
                                   cells, faces, grad.p);
        limiter.calculate_limiters(flow_states.gas.rho(), limiters_.rho, cells,
                                   faces, grad.rho);
        limiter.calculate_limiters(flow_states.vel.x(), limiters_.vx, cells,
                                   faces, grad.vx);
        limiter.calculate_limiters(flow_states.vel.y(), limiters_.vy, cells,
                                   faces, grad.vy);
        limiter.calculate_limiters(flow_states.vel.z(), limiters_.vz, cells,
                                   faces, grad.vz);
    }

    Kokkos::parallel_for(
        "FV::linear_reconstruct", grid.num_interfaces(),
        KOKKOS_LAMBDA(const int i_face) {
            int left_cell = faces.left_cell(i_face);
            bool left_valid = left_cell < num_cells;
            T dx = faces.centre().x(i_face) - cells.centroids().x(left_cell);
            T dy = faces.centre().y(i_face) - cells.centroids().y(left_cell);
            T dz = faces.centre().z(i_face) - cells.centroids().z(left_cell);

            T p_limit = limiter_enabled ? limiters.p(left_cell) : 1.0;
            left.gas.pressure(i_face) =
                linear_interpolate(flow_states.gas.pressure(left_cell), grad.p,
                                   dx, dy, dz, left_cell, p_limit, left_valid);
            T rho_limit = limiter_enabled ? limiters.rho(left_cell) : 1.0;
            left.gas.rho(i_face) =
                linear_interpolate(flow_states.gas.rho(left_cell), grad.rho, dx,
                                   dy, dz, left_cell, rho_limit, left_valid);
            T vx_limit = limiter_enabled ? limiters.vx(left_cell) : 1.0;
            left.vel.x(i_face) =
                linear_interpolate(flow_states.vel.x(left_cell), grad.vx, dx,
                                   dy, dz, left_cell, vx_limit, left_valid);
            T vy_limit = limiter_enabled ? limiters.vy(left_cell) : 1.0;
            left.vel.y(i_face) =
                linear_interpolate(flow_states.vel.y(left_cell), grad.vy, dx,
                                   dy, dz, left_cell, vy_limit, left_valid);
            T vz_limit = limiter_enabled ? limiters.vz(left_cell) : 1.0;
            left.vel.z(i_face) =
                linear_interpolate(flow_states.vel.z(left_cell), grad.vz, dx,
                                   dy, dz, left_cell, vz_limit, left_valid);
            gas_model.update_thermo_from_rhop(left.gas, i_face);

            int right_cell = faces.right_cell(i_face);
            bool right_valid = right_cell < num_cells;
            dx = faces.centre().x(i_face) - cells.centroids().x(right_cell);
            dy = faces.centre().y(i_face) - cells.centroids().y(right_cell);
            dz = faces.centre().z(i_face) - cells.centroids().z(right_cell);
            p_limit = limiter_enabled ? limiters.p(right_cell) : 1.0;
            right.gas.pressure(i_face) = linear_interpolate(
                flow_states.gas.pressure(right_cell), grad.p, dx, dy, dz,
                right_cell, p_limit, right_valid);
            rho_limit = limiter_enabled ? limiters.p(right_cell) : 1.0;
            right.gas.rho(i_face) = linear_interpolate(
                flow_states.gas.rho(right_cell), grad.rho, dx, dy, dz,
                right_cell, rho_limit, right_valid);
            vx_limit = limiter_enabled ? limiters.vx(right_cell) : 1.0;
            right.vel.x(i_face) =
                linear_interpolate(flow_states.vel.x(right_cell), grad.vx, dx,
                                   dy, dz, right_cell, vx_limit, right_valid);
            vy_limit = limiter_enabled ? limiters.vy(right_cell) : 1.0;
            right.vel.y(i_face) =
                linear_interpolate(flow_states.vel.y(right_cell), grad.vy, dx,
                                   dy, dz, right_cell, vy_limit, right_valid);
            vz_limit = limiter_enabled ? limiters.vz(right_cell) : 1.0;
            right.vel.z(i_face) =
                linear_interpolate(flow_states.vel.z(right_cell), grad.vz, dx,
                                   dy, dz, right_cell, vz_limit, right_valid);
            gas_model.update_thermo_from_rhop(right.gas, i_face);
        });
}

template <typename T>
void FiniteVolume<T>::compute_convective_flux(const GridBlock<T>& grid,
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
        "flux::transform_to_global", faces.size(),
        KOKKOS_LAMBDA(const size_t i) {
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

template<typename T>
void FiniteVolume<T>::compute_viscous_flux(const FlowStates<T>& flow_states,
                                           const GridBlock<T>& grid,
                                           const IdealGas<T>& gas_model,
                                           const TransportProperties<T>& trans_prop) {
    grad_calc_.compute_gradients(grid, flow_states.gas.temp(), grad_.temp); 
    if (reconstruction_order_ < 2) {
        grad_calc_.compute_gradients(grid, flow_states.vel.x(), grad_.vx);
    }

    ConservedQuantities<T> flux = flux_;
    Interfaces<T> faces = grid.interfaces();
    FlowStates<T> left = left_;
    FlowStates<T> right = right_;
    GasStates<T> face_gs = face_gs_;
    Kokkos::parallel_for(
        "viscous_flux", faces.size(), 
        KOKKOS_LAMBDA(const size_t i) {
            // compute the viscosity and thermal conductivity at the face
            face_gs.temp(i) = 0.5 * (left.gas.temp(i) + right.gas.temp(i));
            face_gs.pressure(i) = 0.5 * (left.gas.pressure(i) + right.gas.pressure(i));
            gas_model.update_thermo_from_pT(face_gs, i);
            T mu = trans_prop.viscosity(face_gs, gas_model, i);
            T k = trans_prop.thermal_conductivity(face_gs, gas_model, i);

            // Use Hasslebacker formula to average gradients to the face

            // Calculate the viscous fluxes
        });
}

template <typename T>
void FiniteVolume<T>::flux_surface_integral(const GridBlock<T>& grid,
                                            ConservedQuantities<T>& dudt) {
    Cells<T> cells = grid.cells();
    CellFaces<T> cell_faces = grid.cells().faces();
    Interfaces<T> faces = grid.interfaces();
    ConservedQuantities<T> flux = flux_;
    size_t num_cells = grid.num_cells();
    Kokkos::parallel_for(
        "flux_integral", num_cells, KOKKOS_LAMBDA(const size_t cell_i) {
            auto face_ids = cell_faces.face_ids(cell_i);
            T d_mass = 0.0;
            T d_momentum_x = 0.0;
            T d_momentum_y = 0.0;
            T d_momentum_z = 0.0;
            T d_energy = 0.0;
            for (size_t face_i = 0; face_i < face_ids.size(); face_i++) {
                size_t face_id = face_ids(face_i);
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
size_t FiniteVolume<T>::count_bad_cells(const FlowStates<T>& fs,
                                        const size_t num_cells) {
    size_t n_bad_cells = 0;
    Kokkos::parallel_reduce(
        "FiniteVolume::count_bad_cells", num_cells,
        KOKKOS_LAMBDA(const int cell_i, size_t& n_bad_cells_utd) {
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
