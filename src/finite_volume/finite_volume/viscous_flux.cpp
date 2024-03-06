#include <finite_volume/viscous_flux.h>

#include "finite_volume/conserved_quantities.h"
#include "finite_volume/gradient.h"
#include "gas/transport_properties.h"

template <typename T>
ViscousFlux<T>::ViscousFlux(const GridBlock<T>& grid, json config) {
    enabled_ = config.at("enabled");

    if (enabled_) {
        face_fs_ = FlowStates<T>(grid.num_interfaces());

        // we only need viscous gradient info at faces
        face_grad_ = Gradients<T>(grid.num_interfaces(), false, true);
    }
}

template <typename T>
void ViscousFlux<T>::compute_viscous_gradient(const FlowStates<T>& flow_states,
                                              const GridBlock<T>& grid,
                                              Gradients<T>& cell_grad,
                                              WLSGradient<T>& grad_calc) {
    grad_calc.compute_gradients(grid, flow_states.gas.temp(), cell_grad.temp);
    grad_calc.compute_gradients(grid, flow_states.vel.x(), cell_grad.vx);
    grad_calc.compute_gradients(grid, flow_states.vel.y(), cell_grad.vy);
    grad_calc.compute_gradients(grid, flow_states.vel.z(), cell_grad.vz);
}

template <typename T>
void ViscousFlux<T>::compute_viscous_flux(
    const FlowStates<T>& flow_states, const GridBlock<T>& grid,
    const IdealGas<T>& gas_model, const TransportProperties<T>& trans_prop,
    Gradients<T>& cell_grad, WLSGradient<T>& grad_calc, ConservedQuantities<T>& flux) {
    compute_viscous_gradient(flow_states, grid, cell_grad, grad_calc);
    compute_viscous_properties_at_faces(flow_states, grid, gas_model, cell_grad);

    size_t num_faces = grid.num_interfaces();
    Interfaces<T> interfaces = grid.interfaces();
    FlowStates<T> face_fs = face_fs_;
    Gradients<T> grad = face_grad_;
    size_t dim = grid.dim();
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
            T theta_x = vx * tau_xx + vy * tau_xy + vz * tau_xz + k * grad.temp.x(i);
            T theta_y = vx * tau_xy + vy * tau_yy + vz * tau_yz + k * grad.temp.y(i);
            T theta_z = vx * tau_xz + vy * tau_yz + vz * tau_zz + k * grad.temp.z(i);

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
void ViscousFlux<T>::compute_viscous_properties_at_faces(const FlowStates<T>& flow_states,
                                                         const GridBlock<T>& grid,
                                                         const IdealGas<T>& gas_model,
                                                         Gradients<T>& cell_grad) {
    Interfaces<T> faces = grid.interfaces();
    Cells<T> cells = grid.cells();
    FlowStates<T> face_fs = face_fs_;
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
                T ex = cells.centroids().x(right_cell) - cells.centroids().x(left_cell);
                T ey = cells.centroids().y(right_cell) - cells.centroids().y(left_cell);
                T ez = cells.centroids().z(right_cell) - cells.centroids().z(left_cell);
                T len_e = Kokkos::sqrt(ex * ex + ey * ey + ez * ez);
                T ehatx = ex / len_e;
                T ehaty = ey / len_e;
                T ehatz = ez / len_e;
                T nx = faces.norm().x(i);
                T ny = faces.norm().y(i);
                T nz = faces.norm().z(i);
                T ehat_dot_n = ehatx * nx + ehaty * ny + ehatz * nz;

                // vx
                T avg_grad_x =
                    0.5 * (cell_grad.vx.x(left_cell) + cell_grad.vx.x(right_cell));
                T avg_grad_y =
                    0.5 * (cell_grad.vx.y(left_cell) + cell_grad.vx.y(right_cell));
                T avg_grad_z =
                    0.5 * (cell_grad.vx.z(left_cell) + cell_grad.vx.z(right_cell));
                T avgdotehat =
                    avg_grad_x * ehatx + avg_grad_y * ehaty + avg_grad_z * ehatz;
                T correction = avgdotehat - (flow_states.vel.x(right_cell) -
                                             flow_states.vel.x(left_cell)) /
                                                len_e;
                face_grad.vx.x(i) = avg_grad_x - correction * nx / ehat_dot_n;
                face_grad.vx.y(i) = avg_grad_y - correction * ny / ehat_dot_n;
                face_grad.vx.z(i) = avg_grad_z - correction * nz / ehat_dot_n;

                // vy
                avg_grad_x =
                    0.5 * (cell_grad.vy.x(left_cell) + cell_grad.vy.x(right_cell));
                avg_grad_y =
                    0.5 * (cell_grad.vy.y(left_cell) + cell_grad.vy.y(right_cell));
                avg_grad_z =
                    0.5 * (cell_grad.vy.z(left_cell) + cell_grad.vy.z(right_cell));
                avgdotehat = avg_grad_x * ehatx + avg_grad_y * ehaty + avg_grad_z * ehatz;
                correction = avgdotehat - (flow_states.vel.y(right_cell) -
                                           flow_states.vel.y(left_cell)) /
                                              len_e;
                face_grad.vy.x(i) = avg_grad_x - correction * nx / ehat_dot_n;
                face_grad.vy.y(i) = avg_grad_y - correction * ny / ehat_dot_n;
                face_grad.vy.z(i) = avg_grad_z - correction * nz / ehat_dot_n;

                // vz
                avg_grad_x =
                    0.5 * (cell_grad.vz.x(left_cell) + cell_grad.vz.x(right_cell));
                avg_grad_y =
                    0.5 * (cell_grad.vz.y(left_cell) + cell_grad.vz.y(right_cell));
                avg_grad_z =
                    0.5 * (cell_grad.vz.z(left_cell) + cell_grad.vz.z(right_cell));
                avgdotehat = avg_grad_x * ehatx + avg_grad_y * ehaty + avg_grad_z * ehatz;
                correction = avgdotehat - (flow_states.vel.z(right_cell) -
                                           flow_states.vel.z(left_cell)) /
                                              len_e;
                face_grad.vz.x(i) = avg_grad_x - correction * nx / ehat_dot_n;
                face_grad.vz.y(i) = avg_grad_y - correction * ny / ehat_dot_n;
                face_grad.vz.z(i) = avg_grad_z - correction * nz / ehat_dot_n;

                // temperature
                avg_grad_x =
                    0.5 * (cell_grad.temp.x(left_cell) + cell_grad.temp.x(right_cell));
                avg_grad_y =
                    0.5 * (cell_grad.temp.y(left_cell) + cell_grad.temp.y(right_cell));
                avg_grad_z =
                    0.5 * (cell_grad.temp.z(left_cell) + cell_grad.temp.z(right_cell));
                avgdotehat = avg_grad_x * ehatx + avg_grad_y * ehaty + avg_grad_z * ehatz;
                correction = avgdotehat - (flow_states.gas.temp(right_cell) -
                                           flow_states.gas.temp(left_cell)) /
                                              len_e;
                face_grad.temp.x(i) = avg_grad_x - correction * nx / ehat_dot_n;
                face_grad.temp.y(i) = avg_grad_y - correction * ny / ehat_dot_n;
                face_grad.temp.z(i) = avg_grad_z - correction * nz / ehat_dot_n;
            }
            // set the gas state at the interface
            face_fs.gas.temp(i) = 0.5 * (flow_states.gas.temp(left_cell) +
                                         flow_states.gas.temp(right_cell));
            face_fs.gas.pressure(i) = 0.5 * (flow_states.gas.pressure(left_cell) +
                                             flow_states.gas.pressure(right_cell));
            gas_model.update_thermo_from_pT(face_fs.gas, i);

            face_fs.vel.x(i) =
                0.5 * (flow_states.vel.x(left_cell) + flow_states.vel.x(right_cell));
            face_fs.vel.y(i) =
                0.5 * (flow_states.vel.y(left_cell) + flow_states.vel.y(right_cell));
            face_fs.vel.z(i) =
                0.5 * (flow_states.vel.z(left_cell) + flow_states.vel.z(right_cell));
        });
}

template class ViscousFlux<double>;
