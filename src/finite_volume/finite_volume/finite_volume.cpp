#include <finite_volume/boundaries/boundary.h>
#include <finite_volume/conserved_quantities.h>
#include <finite_volume/finite_volume.h>
#include <finite_volume/flux_calc.h>

#include <stdexcept>

#include "finite_volume/gradient.h"
#include "gas/transport_properties.h"

template <typename T>
FiniteVolume<T>::FiniteVolume(const GridBlock<T>& grid, json config)
    : left_(FlowStates<T>(grid.num_interfaces())),
      right_(FlowStates<T>(grid.num_interfaces())),
      flux_(ConservedQuantities<T>(grid.num_interfaces(), grid.dim())) {
    dim_ = grid.dim();

    json convective_flux_config = config.at("convective_flux");
    json viscous_flux_config = config.at("viscous_flux");
    viscous_ = viscous_flux_config.at("enabled");

    // allocate memory for gradients
    if (viscous_ || reconstruction_order_ > 1) {
        bool convective_grad = reconstruction_order_ > 1;
        grad_calc_ = WLSGradient<T>(grid);
        cell_grad_ = Gradients<T>(grid.num_cells(), convective_grad, viscous_);

        if (viscous_) {
            face_fs_ = FlowStates<T>(grid.num_interfaces());

            // we only need viscous gradient info at faces
            face_grad_ = Gradients<T>(grid.num_interfaces(), false, true);
        }
    }

    // set the flux calculator
    flux_calculator_ = flux_calculator_from_string(
        convective_flux_config.at("flux_calculator"));

    // set up reconstruction limiters for convective flux
    reconstruction_order_ = convective_flux_config.at("reconstruction_order");
    if (reconstruction_order_ > 1) {
        limiter_ = Limiter<T>(convective_flux_config);
        if (limiter_.enabled()) {
            limiters_ = LimiterValues<T>(grid.num_cells());
        }
    }

    // set up boundary conditions
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
        apply_pre_viscous_grad_bc(flow_state, grid);
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
    FlowStates<T> face_fs = face_fs_;
    bool viscous = viscous_;
    // IdealGas<T> gas_model = gas_model_;

    double dt;
    Kokkos::parallel_reduce(
        "FV::signal_frequency", num_cells,
        KOKKOS_LAMBDA(const size_t cell_i, double& dt_utd) {
            auto cell_face_ids = cell_interfaces.face_ids(cell_i);

            T vx = flow_state.vel.x(cell_i);
            T vy = flow_state.vel.y(cell_i);
            T vz = flow_state.vel.z(cell_i);

            T spectral_radii_c = 0.0;
            T spectral_radii_v = 0.0;
            T volume = cells.volume(cell_i);
            for (size_t face_idx = 0; face_idx < cell_face_ids.size();
                 face_idx++) {
                size_t i_face = cell_face_ids(face_idx);
                T area = interfaces.area(i_face);
                T dot = vx * interfaces.norm().x(i_face) +
                        vy * interfaces.norm().y(i_face) +
                        vz * interfaces.norm().z(i_face);
                T sig_vel = Kokkos::fabs(dot) +
                            gas_model.speed_of_sound(flow_state.gas, cell_i);
                spectral_radii_c += sig_vel * area;

                if (viscous) {
                    T gamma = gas_model.gamma();
                    T mu = trans_prop.viscosity(face_fs.gas, gas_model, i_face);
                    T k = trans_prop.thermal_conductivity(face_fs.gas,
                                                          gas_model, i_face);
                    T rho = face_fs.gas.rho(i_face);
                    T Pr = mu * gas_model.Cp() / k;
                    T tmp = (gamma / rho) * (mu / Pr) * area * area;
                    spectral_radii_v += tmp / volume;
                }
            }
            T local_dt = volume / (spectral_radii_c + 4 * spectral_radii_v);
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
void FiniteVolume<T>::apply_pre_viscous_grad_bc(FlowStates<T>& fs,
                                                const GridBlock<T>& grid) {
    for (size_t i = 0; i < bcs_.size(); i++) {
        std::shared_ptr<BoundaryCondition<T>> bc = bcs_[i];
        Field<size_t> bc_faces = bc_interfaces_[i];
        bc->apply_pre_viscous_grad(fs, grid, bc_faces);
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
    grad_calc_.compute_gradients(grid, flow_states.gas.pressure(),
                                 cell_grad_.p);
    grad_calc_.compute_gradients(grid, flow_states.gas.rho(), cell_grad_.rho);
    grad_calc_.compute_gradients(grid, flow_states.vel.x(), cell_grad_.vx);
    grad_calc_.compute_gradients(grid, flow_states.vel.y(), cell_grad_.vy);
    grad_calc_.compute_gradients(grid, flow_states.vel.z(), cell_grad_.vz);

    auto cells = grid.cells();
    auto faces = grid.interfaces();
    auto left = left_;
    auto right = right_;
    auto grad = cell_grad_;
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
        case FluxCalculator::Ldfss:
            ldfss(left_, right_, flux_, gas_model, dim_ == 3);
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

template <typename T>
void FiniteVolume<T>::compute_viscous_properties_at_faces(
    const FlowStates<T>& flow_states, const GridBlock<T>& grid,
    const IdealGas<T>& gas_model) {
    grad_calc_.compute_gradients(grid, flow_states.gas.temp(), cell_grad_.temp);
    grad_calc_.compute_gradients(grid, flow_states.vel.x(), cell_grad_.vx);
    grad_calc_.compute_gradients(grid, flow_states.vel.y(), cell_grad_.vy);
    grad_calc_.compute_gradients(grid, flow_states.vel.z(), cell_grad_.vz);

    ConservedQuantities<T> flux = flux_;
    Interfaces<T> faces = grid.interfaces();
    Cells<T> cells = grid.cells();
    FlowStates<T> left = left_;
    FlowStates<T> right = right_;
    FlowStates<T> face_fs = face_fs_;
    Gradients<T> cell_grad = cell_grad_;
    Gradients<T> face_grad = face_grad_;
    size_t num_cells = grid.num_cells();
    Kokkos::parallel_for(
        "viscous_properties", faces.size(), KOKKOS_LAMBDA(const size_t i) {
            size_t left_cell = faces.left_cell(i);
            size_t right_cell = faces.right_cell(i);
            bool left_valid = left_cell < num_cells;
            bool right_valid = right_cell < num_cells;
            if (!left_valid || !right_valid) {
                size_t interior_cell = (left_valid) ? left_cell : right_cell;

                // copy gradients to the face
                face_grad.temp.x(i) = cell_grad.temp.x(interior_cell);
                face_grad.temp.y(i) = cell_grad.temp.y(interior_cell);
                face_grad.temp.z(i) = cell_grad.temp.z(interior_cell);

                face_grad.vx.x(i) = cell_grad.vx.x(interior_cell);
                face_grad.vx.y(i) = cell_grad.vx.y(interior_cell);
                face_grad.vx.z(i) = cell_grad.vx.z(interior_cell);

                face_grad.vy.x(i) = cell_grad.vy.x(interior_cell);
                face_grad.vy.y(i) = cell_grad.vy.y(interior_cell);
                face_grad.vy.z(i) = cell_grad.vy.z(interior_cell);

                face_grad.vz.x(i) = cell_grad.vz.x(interior_cell);
                face_grad.vz.y(i) = cell_grad.vz.y(interior_cell);
                face_grad.vz.z(i) = cell_grad.vz.z(interior_cell);
            } else {
                // Use Hasselbacher formula to average gradients to the face

                // vector from cell centre to cell centre
                T ex = cells.centroids().x(right_cell) -
                       cells.centroids().x(left_cell);
                T ey = cells.centroids().y(right_cell) -
                       cells.centroids().y(left_cell);
                T ez = cells.centroids().z(right_cell) -
                       cells.centroids().z(left_cell);
                T len_e = Kokkos::sqrt(ex * ex + ey * ey + ez * ez);
                T ehatx = ex / len_e;
                T ehaty = ey / len_e;
                T ehatz = ez / len_e;
                T nx = faces.norm().x(i);
                T ny = faces.norm().y(i);
                T nz = faces.norm().z(i);
                T ehat_dot_n = ehatx * nx + ehaty * ny + ehatz * nz;

                // vx
                T avg_grad_x = 0.5 * (cell_grad.vx.x(left_cell) +
                                      cell_grad.vx.x(right_cell));
                T avg_grad_y = 0.5 * (cell_grad.vx.y(left_cell) +
                                      cell_grad.vx.y(right_cell));
                T avg_grad_z = 0.5 * (cell_grad.vx.z(left_cell) +
                                      cell_grad.vx.z(right_cell));
                T avgdotehat = avg_grad_x * ehatx + avg_grad_y * ehaty +
                               avg_grad_z * ehatz;
                T correction =
                    avgdotehat - (right.vel.x(i) - left.vel.x(i)) / len_e;
                face_grad.vx.x(i) = avg_grad_x - correction * nx / ehat_dot_n;
                face_grad.vx.y(i) = avg_grad_y - correction * ny / ehat_dot_n;
                face_grad.vx.z(i) = avg_grad_z - correction * nz / ehat_dot_n;

                // vy
                avg_grad_x = 0.5 * (cell_grad.vy.x(left_cell) +
                                    cell_grad.vy.x(right_cell));
                avg_grad_y = 0.5 * (cell_grad.vy.y(left_cell) +
                                    cell_grad.vy.y(right_cell));
                avg_grad_z = 0.5 * (cell_grad.vy.z(left_cell) +
                                    cell_grad.vy.z(right_cell));
                avgdotehat = avg_grad_x * ehatx + avg_grad_y * ehaty +
                             avg_grad_z * ehatz;
                correction =
                    avgdotehat - (right.vel.y(i) - left.vel.y(i)) / len_e;
                face_grad.vy.x(i) = avg_grad_x - correction * nx / ehat_dot_n;
                face_grad.vy.y(i) = avg_grad_y - correction * ny / ehat_dot_n;
                face_grad.vy.z(i) = avg_grad_z - correction * nz / ehat_dot_n;

                // vz
                avg_grad_x = 0.5 * (cell_grad.vz.x(left_cell) +
                                    cell_grad.vz.x(right_cell));
                avg_grad_y = 0.5 * (cell_grad.vz.y(left_cell) +
                                    cell_grad.vz.y(right_cell));
                avg_grad_z = 0.5 * (cell_grad.vz.z(left_cell) +
                                    cell_grad.vz.z(right_cell));
                avgdotehat = avg_grad_x * ehatx + avg_grad_y * ehaty +
                             avg_grad_z * ehatz;
                correction =
                    avgdotehat - (right.vel.z(i) - left.vel.z(i)) / len_e;
                face_grad.vz.x(i) = avg_grad_x - correction * nx / ehat_dot_n;
                face_grad.vz.y(i) = avg_grad_y - correction * ny / ehat_dot_n;
                face_grad.vz.z(i) = avg_grad_z - correction * nz / ehat_dot_n;

                // temperature
                avg_grad_x = 0.5 * (cell_grad.temp.x(left_cell) +
                                    cell_grad.temp.x(right_cell));
                avg_grad_y = 0.5 * (cell_grad.temp.y(left_cell) +
                                    cell_grad.temp.y(right_cell));
                avg_grad_z = 0.5 * (cell_grad.temp.z(left_cell) +
                                    cell_grad.temp.z(right_cell));
                avgdotehat = avg_grad_x * ehatx + avg_grad_y * ehaty +
                             avg_grad_z * ehatz;
                correction =
                    avgdotehat - (right.gas.temp(i) - left.gas.temp(i)) / len_e;
                face_grad.temp.x(i) = avg_grad_x - correction * nx / ehat_dot_n;
                face_grad.temp.y(i) = avg_grad_y - correction * ny / ehat_dot_n;
                face_grad.temp.z(i) = avg_grad_z - correction * nz / ehat_dot_n;
            }
            // set the gas state at the interface
            face_fs.gas.temp(i) = 0.5 * (flow_states.gas.temp(left_cell) +
                                         flow_states.gas.temp(right_cell));
            face_fs.gas.pressure(i) =
                0.5 * (flow_states.gas.pressure(left_cell) +
                       flow_states.gas.pressure(right_cell));
            gas_model.update_thermo_from_pT(face_fs.gas, i);

            face_fs.vel.x(i) = 0.5 * (flow_states.vel.x(left_cell) +
                                      flow_states.vel.x(right_cell));
            face_fs.vel.y(i) = 0.5 * (flow_states.vel.y(left_cell) +
                                      flow_states.vel.y(right_cell));
            face_fs.vel.z(i) = 0.5 * (flow_states.vel.z(left_cell) +
                                      flow_states.vel.z(right_cell));
        });
}

template <typename T>
void FiniteVolume<T>::compute_viscous_flux(
    const FlowStates<T>& flow_states, const GridBlock<T>& grid,
    const IdealGas<T>& gas_model, const TransportProperties<T>& trans_prop) {
    compute_viscous_properties_at_faces(flow_states, grid, gas_model);

    size_t num_faces = grid.num_interfaces();
    Interfaces<T> interfaces = grid.interfaces();
    FlowStates<T> face_fs = face_fs_;
    ConservedQuantities<T> flux = flux_;
    Gradients<T> grad = face_grad_;
    size_t dim = dim_;
    Kokkos::parallel_for(
        "viscous_flux", num_faces, KOKKOS_LAMBDA(const size_t i) {
            // transport properties at the face
            T mu = trans_prop.viscosity(face_fs.gas, gas_model, i);
            T k = trans_prop.thermal_conductivity(face_fs.gas, gas_model, i);
            T lambda = -2.0 / 3.0 * mu;

            // compute the viscous fluxes
            T bulk = lambda * (grad.vx.x(i) + grad.vy.y(i) + grad.vz.z(i));
            T tau_xx = 2.0 * mu * grad.vx.x(i) + bulk;
            T tau_yy = 2.0 * mu * grad.vy.y(i) + bulk;
            T tau_zz = 2.0 * mu * grad.vz.z(i) + bulk;
            T tau_xy = mu * (grad.vx.y(i) + grad.vy.x(i));
            T tau_xz = mu * (grad.vx.z(i) + grad.vz.x(i));
            T tau_yz = mu * (grad.vy.z(i) + grad.vz.y(i));

            T vx = face_fs.vel.x(i);
            T vy = face_fs.vel.y(i);
            T vz = face_fs.vel.z(i);
            T theta_x =
                vx * tau_xx + vy * tau_xy + vz * tau_xz + k * grad.temp.x(i);
            T theta_y =
                vx * tau_xy + vy * tau_yy + vz * tau_yz + k * grad.temp.y(i);
            T theta_z =
                vx * tau_xz + vy * tau_yz + vz * tau_zz + k * grad.temp.z(i);

            T nx = interfaces.norm().x(i);
            T ny = interfaces.norm().y(i);
            T nz = interfaces.norm().z(i);
            flux.momentum_x(i) -= nx * tau_xx + ny * tau_xy + nz * tau_xz;
            flux.momentum_y(i) -= nx * tau_xy + ny * tau_yy + nz * tau_yz;
            if (dim == 3) {
                flux.momentum_z(i) -= nx * tau_xz + ny * tau_yz + nz * tau_zz;
            }
            flux.energy(i) -= nx * theta_x + ny * theta_y + nz * theta_z;
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
