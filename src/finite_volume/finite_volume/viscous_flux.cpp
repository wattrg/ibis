#include <finite_volume/viscous_flux.h>

#include "finite_volume/conserved_quantities.h"
#include "finite_volume/gradient.h"
#include "gas/transport_properties.h"

template <typename T>
struct ViscousProperties {
    FlowState<T> flow;
    Vector3<T> grad_temp;
    Vector3<T> grad_vx;
    Vector3<T> grad_vy;
    Vector3<T> grad_vz;
};

KOKKOS_FUNCTION
template <typename T>
ViscousProperties<T> compute_viscous_properties_at_faces(const FlowStates<T>& flow_states,
                                                         const Interfaces<T>& faces,
                                                         const Cells<T>& cells,
                                                         const IdealGas<T>& gas_model,
                                                         const Gradients<T>& cell_grad,
                                                         const size_t num_cells,
                                                         const size_t face_i) {
    // Interfaces<T> faces = grid.interfaces();
    // Cells<T> cells = grid.cells();
    // size_t num_cells = grid.num_cells();
    // FlowStates<T> face_fs = face_fs_;
    // Gradients<T> face_grad = face_grad_;

    ViscousProperties<T> props;
    size_t left_cell = faces.left_cell(face_i);
    size_t right_cell = faces.right_cell(face_i);
    bool left_valid = left_cell < num_cells;
    bool right_valid = right_cell < num_cells;
    if (!left_valid || !right_valid) {
        size_t interior_cell = (left_valid) ? left_cell : right_cell;

        // copy gradients to the face
        props.grad_temp.x = cell_grad.temp.x(interior_cell);
        props.grad_temp.y = cell_grad.temp.y(interior_cell);
        props.grad_temp.z = cell_grad.temp.z(interior_cell);

        props.grad_vx.x = cell_grad.vx.x(interior_cell);
        props.grad_vx.y = cell_grad.vx.y(interior_cell);
        props.grad_vx.z = cell_grad.vx.z(interior_cell);

        props.grad_vy.x = cell_grad.vy.x(interior_cell);
        props.grad_vy.y = cell_grad.vy.y(interior_cell);
        props.grad_vy.z = cell_grad.vy.z(interior_cell);

        props.grad_vz.x = cell_grad.vz.x(interior_cell);
        props.grad_vz.y = cell_grad.vz.y(interior_cell);
        props.grad_vz.z = cell_grad.vz.z(interior_cell);
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
        T nx = faces.norm().x(face_i);
        T ny = faces.norm().y(face_i);
        T nz = faces.norm().z(face_i);
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
        props.grad_vx.x = avg_grad_x - correction * nx / ehat_dot_n;
        props.grad_vx.y = avg_grad_y - correction * ny / ehat_dot_n;
        props.grad_vx.z = avg_grad_z - correction * nz / ehat_dot_n;

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
        props.grad_vy.x = avg_grad_x - correction * nx / ehat_dot_n;
        props.grad_vy.y = avg_grad_y - correction * ny / ehat_dot_n;
        props.grad_vy.z = avg_grad_z - correction * nz / ehat_dot_n;

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
        props.grad_vz.x = avg_grad_x - correction * nx / ehat_dot_n;
        props.grad_vz.y = avg_grad_y - correction * ny / ehat_dot_n;
        props.grad_vz.z = avg_grad_z - correction * nz / ehat_dot_n;

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
        props.grad_temp.x = avg_grad_x - correction * nx / ehat_dot_n;
        props.grad_temp.y = avg_grad_y - correction * ny / ehat_dot_n;
        props.grad_temp.z = avg_grad_z - correction * nz / ehat_dot_n;
    }
    // set the gas state at the interface
    props.flow.gas_state.temp = 0.5 * (flow_states.gas.temp(left_cell) +
                                 flow_states.gas.temp(right_cell));
    props.flow.gas_state.pressure = 0.5 * (flow_states.gas.pressure(left_cell) +
                                     flow_states.gas.pressure(right_cell));
    gas_model.update_thermo_from_pT(props.flow.gas_state);

    props.flow.velocity.x =
        0.5 * (flow_states.vel.x(left_cell) + flow_states.vel.x(right_cell));
    props.flow.velocity.y =
        0.5 * (flow_states.vel.y(left_cell) + flow_states.vel.y(right_cell));
    props.flow.velocity.z =
        0.5 * (flow_states.vel.z(left_cell) + flow_states.vel.z(right_cell));
    return props;
}

template <typename T>
ViscousFlux<T>::ViscousFlux(const GridBlock<T>& grid, json config) {
    enabled_ = config.at("enabled");
    signal_factor_ = config.at("signal_factor");
    (void) grid;

    if (enabled_) {
        face_fs_ = FlowStates<T>(grid.num_interfaces());

        // we only need viscous gradient info at faces
        // face_grad_ =
            // Gradients<T>(grid.num_interfaces(), false, false, false, false, true);
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
    // compute_viscous_properties_at_faces(flow_states, grid, gas_model, cell_grad);

    size_t num_faces = grid.num_interfaces();
    Interfaces<T> interfaces = grid.interfaces();
    Cells<T> cells = grid.cells();
    const size_t num_cells = grid.num_cells();
    // FlowStates<T> face_fs = face_fs_;
    // Gradients<T> grad = face_grad_;
    size_t dim = grid.dim();
    Kokkos::parallel_for(
        "viscous_flux", num_faces, KOKKOS_LAMBDA(const size_t i) {
            auto props = compute_viscous_properties_at_faces(flow_states, interfaces,
                                                             cells, gas_model, cell_grad, 
                                                             num_cells, i);
            face_fs_.set_flow_state(props.flow, i);

            
            // transport properties at the face
            T mu = trans_prop.viscosity(props.flow.gas_state, gas_model);
            T k = trans_prop.thermal_conductivity(props.flow.gas_state, gas_model);
            T lambda = -2.0 / 3.0 * mu;

            // compute the viscous fluxes
            T bulk = lambda * (props.grad_vx.x + props.grad_vy.y + props.grad_vz.z);
            T tau_xx = 2.0 * mu * props.grad_vx.x + bulk;
            T tau_yy = 2.0 * mu * props.grad_vy.y + bulk;
            T tau_zz = 2.0 * mu * props.grad_vz.z + bulk;
            T tau_xy = mu * (props.grad_vx.y + props.grad_vy.x);
            T tau_xz = mu * (props.grad_vx.z + props.grad_vz.x);
            T tau_yz = mu * (props.grad_vy.z + props.grad_vz.y);

            T vx = props.flow.velocity.x;
            T vy = props.flow.velocity.y;
            T vz = props.flow.velocity.z;
            T theta_x = vx * tau_xx + vy * tau_xy + vz * tau_xz + k * props.grad_temp.x;
            T theta_y = vx * tau_xy + vy * tau_yy + vz * tau_yz + k * props.grad_temp.y;
            T theta_z = vx * tau_xz + vy * tau_yz + vz * tau_zz + k * props.grad_temp.z;

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


template class ViscousFlux<double>;
