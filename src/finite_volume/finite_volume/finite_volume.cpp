#include <finite_volume/boundaries/boundary.h>
#include <finite_volume/convective_flux.h>
#include <finite_volume/finite_volume.h>
#include <finite_volume/flux_calc.h>
#include <parallel/parallel.h>
#include <util/numeric_types.h>

#include "gas/transport_properties.h"

template <typename T, class MemModel>
FiniteVolume<T, MemModel>::FiniteVolume(GridBlock<MemModel, T>& grid, json config) {
    dim_ = grid.dim();

    json convective_flux_config = config.at("convective_flux");
    json viscous_flux_config = config.at("viscous_flux");

    // allocate memory for fluxes
    flux_ = ConservedQuantities<T>(grid.num_interfaces(), grid.dim());

    // set up the convective flux
    convective_flux_ = ConvectiveFlux<T, MemModel>(grid, convective_flux_config);

    // flow states on the interfaces
    face_fs_ = FlowStates<T>(grid.num_interfaces());

    // set up the viscous flux
    viscous_flux_ = ViscousFlux<T, MemModel>(grid, face_fs_, viscous_flux_config);

    // allocate memory for gradients
    size_t reconstruction_order = convective_flux_.reconstruction_order();
    bool viscous = viscous_flux_.enabled();
    if (viscous || reconstruction_order > 1) {
        grid.allocate_gradient_weights();
        const RequiredGradients grads = convective_flux_.required_gradients();
        cell_grad_ = Gradients<T>(grid.num_cells(), grads.pressure, grads.temp, grads.u,
                                  grads.rho, viscous);
    }

    // set up physical boundary conditions
    std::vector<std::string> boundary_tags = grid.boundary_tags();
    json boundaries_config = config.at("grid").at("boundaries");
    for (size_t bi = 0; bi < boundary_tags.size(); bi++) {
        // build the actual boundary condition
        json boundary_config = boundaries_config.at(boundary_tags[bi]);
        std::shared_ptr<BoundaryCondition<T, MemModel>> boundary(
            new BoundaryCondition<T, MemModel>(boundary_config));
        bcs_.push_back(boundary);

        // the faces associated with this boundary
        bc_interfaces_.push_back(grid.boundary_faces(boundary_tags[bi]));
    }

    // setup internal boundaries
    size_t num_flow_vars = (grid.dim() == 3) ? 5 : 4;
    size_t num_grads = cell_grad_.num_grads();
    for (size_t block_i = 0; block_i < grid.other_blocks().size(); block_i++) {
        size_t other_block = grid.other_block(block_i);
        size_t num_cells_on_boundary =
            grid.internal_boundary_ghost_cells(other_block).size();
        flow_state_comm_.push_back(Ibis::SymmetricComm<MemModel, T>(
            other_block, num_cells_on_boundary * num_flow_vars));
        gradient_comm_.push_back(Ibis::SymmetricComm<MemModel, T>(
            other_block, num_cells_on_boundary * num_grads));
    }
}

template <typename T, class MemModel>
size_t FiniteVolume<T, MemModel>::compute_dudt(
    FlowStates<T>& flow_state, Vector3s<T> vertex_vel, const ConservedQuantities<T>& cq,
    GridBlock<MemModel, T>& grid, ConservedQuantities<T>& dudt, IdealGas<T>& gas_model,
    TransportProperties<T>& trans_prop, bool allow_reconstruction) {
    if constexpr (std::is_same<MemModel, Mpi>::value) {
        transfer_internal_flowstates(flow_state, grid, gas_model);
    }
    apply_pre_reconstruction_bc(flow_state, grid, gas_model, trans_prop);
    if (grid.moving()) {
        grid.compute_grid_motion(flow_state, vertex_vel);
    }

    convective_flux_.compute_convective_flux(flow_state, grid, gas_model, cell_grad_,
                                             grid.grad_calc(), flux_,
                                             allow_reconstruction);

    apply_post_convective_flux_bc(flow_state, grid, gas_model, trans_prop);

    if (viscous_flux_.enabled()) {
        apply_pre_viscous_grad_bc(flow_state, grid, gas_model, trans_prop);
        viscous_flux_.compute_viscous_flux(flow_state, grid, gas_model, trans_prop,
                                           cell_grad_, grid.grad_calc(), flux_);
    }

    flux_surface_integral(grid, dudt);

    if (grid.moving()) {
        apply_geometric_conservation_law(cq, grid, dudt);
    }
    return 0;
}

template <typename T, class MemModel>
void FiniteVolume<T, MemModel>::transfer_internal_flowstates(
    FlowStates<T>& fs, const GridBlock<MemModel, T>& grid, const IdealGas<T>& gas_model) {
    (void)gas_model;
    size_t num_vars = (grid.dim() == 3) ? 5 : 4;
    size_t dim = grid.dim();

    // Step 0: Post a receive so that we can wait for incoming data
    for (auto& comm : flow_state_comm_) {
        comm.expect_receive();
    }
    
    // Step 1: pack send buffers
    for (size_t boundary_i = 0; boundary_i < grid.other_blocks().size(); boundary_i++) {
        size_t other_block = grid.other_block(boundary_i);
        Ibis::SymmetricComm<MemModel, T> comm = flow_state_comm_[boundary_i];
        auto cells_to_pack = grid.internal_boundary_cells(other_block);
        auto buffer = comm.send_buf();

        // the parallel work of packing the data
        Ibis::parallel_for(
            "FV::pack_send_buffer", cells_to_pack.size(),
            KOKKOS_LAMBDA(const size_t cell_i) {
                size_t cell_to_pack = cells_to_pack(cell_i);
                size_t start_index = cell_i * num_vars;
                buffer(start_index + 0) = fs.gas.rho(cell_to_pack);
                buffer(start_index + 1) = fs.gas.pressure(cell_to_pack);
                buffer(start_index + 2) = fs.vel.x(cell_to_pack);
                buffer(start_index + 3) = fs.vel.y(cell_to_pack);
                if (dim == 3) {
                    buffer(start_index + 4) = fs.vel.z(cell_to_pack);
                }
            });
    }

    // Step 2: transfer data
    for (auto& comm : flow_state_comm_) {
        comm.send();
        comm.receive();
    }

    // Step 3: unpack receive buffers
    for (size_t boundary_i = 0; boundary_i < grid.other_blocks().size(); boundary_i++) {
        size_t other_block = grid.other_block(boundary_i);
        Ibis::SymmetricComm<MemModel, T> comm = flow_state_comm_[boundary_i];
        auto cells_to_unpack_to = grid.internal_boundary_ghost_cells(other_block);
        auto buffer = comm.recv_buf();

        // unpack the buffer
        Ibis::parallel_for(
            "FV::unpack_recv_buffer", cells_to_unpack_to.size(), KOKKOS_LAMBDA(const size_t cell_i) {
                size_t cell_to_unpack_to = cells_to_unpack_to(cell_i);
                size_t start_index = cell_i * num_vars;
                fs.gas.rho(cell_to_unpack_to) = buffer(start_index + 0);
                fs.gas.pressure(cell_to_unpack_to) = buffer(start_index + 1);
                fs.vel.x(cell_to_unpack_to) = buffer(start_index + 2);
                fs.vel.y(cell_to_unpack_to) = buffer(start_index + 3);
                if (dim == 3) {
                    fs.vel.z(cell_to_unpack_to) = buffer(start_index + 4);
                }
            }  
        );
    }
}

template <typename T, class MemModel>
size_t FiniteVolume<T, MemModel>::compute_dudt(FlowStates<T>& flow_state,
                                               GridBlock<MemModel, T>& grid,
                                               ConservedQuantities<T>& dudt,
                                               IdealGas<T>& gas_model,
                                               TransportProperties<T>& trans_prop,
                                               bool allow_reconstruction) {
    Vector3s<T> vertex_vel_temp;
    const ConservedQuantities<T> cq_temp;
    return compute_dudt(flow_state, vertex_vel_temp, cq_temp, grid, dudt, gas_model,
                        trans_prop, allow_reconstruction);
}

template <typename T, class MemModel>
void FiniteVolume<T, MemModel>::apply_geometric_conservation_law(
    const ConservedQuantities<T>& cq, const GridBlock<MemModel, T>& grid,
    ConservedQuantities<T>& dudt) {
    size_t num_cells = grid.num_cells();
    Cells<T> cells = grid.cells();
    CellFaces<T> cell_faces = grid.cells().faces();
    Interfaces<T> faces = grid.interfaces();
    Vector3s<T> face_vel = grid.face_vel();
    Vector3s<T> face_norm = faces.norm();
    Kokkos::parallel_for(
        "FV::GCL", num_cells, KOKKOS_LAMBDA(const size_t cell_i) {
            auto face_ids = cell_faces.face_ids(cell_i);
            T dVdt = T(0.0);
            for (size_t face_i = 0; face_i < face_ids.size(); face_i++) {
                size_t face_id = face_ids(face_i);
                T area = faces.area(face_id) * cell_faces.outsigns(cell_i)(face_i);
                Vector3<T> vel = face_vel.vector(face_id);
                Vector3<T> norm = face_norm.vector(face_id);
                dVdt += (vel.x * norm.x + vel.y * norm.y + vel.z * norm.z) * area;
            }
            T V = cells.volume(cell_i);
            dudt.mass(cell_i) -= cq.mass(cell_i) * dVdt / V;
            dudt.momentum_x(cell_i) -= cq.momentum_x(cell_i) * dVdt / V;
            dudt.momentum_y(cell_i) -= cq.momentum_y(cell_i) * dVdt / V;
            if (cq.dim() == 3) {
                dudt.momentum_z(cell_i) -= cq.momentum_z(cell_i) * dVdt / V;
            }
            dudt.energy(cell_i) -= cq.energy(cell_i) * dVdt / V;
        });
}

template <typename T, class MemModel>
Ibis::real FiniteVolume<T, MemModel>::estimate_dt(const FlowStates<T>& flow_state,
                                                  GridBlock<MemModel, T>& grid,
                                                  IdealGas<T>& gas_model,
                                                  TransportProperties<T>& trans_prop) {
    (void)trans_prop;
    size_t num_cells = grid.num_cells();
    CellFaces<T> cell_interfaces = grid.cells().faces();
    Interfaces<T> interfaces = grid.interfaces();
    Cells<T> cells = grid.cells();
    FlowStates<T> face_fs = viscous_flux_.face_fs();
    bool viscous = viscous_flux_.enabled();
    Ibis::real viscous_signal_factor = viscous_flux_.signal_factor();
    // IdealGas<T> gas_model = gas_model_;

    return Ibis::parallel_reduce<Min<Ibis::real>, Ibis::DefaultMemModel>(
        "FV::signal_frequency", num_cells,
        KOKKOS_LAMBDA(const size_t cell_i, Ibis::real& dt_utd) {
            auto cell_face_ids = cell_interfaces.face_ids(cell_i);

            T vx = flow_state.vel.x(cell_i);
            T vy = flow_state.vel.y(cell_i);
            T vz = flow_state.vel.z(cell_i);

            T spectral_radii_c = 0.0;
            T spectral_radii_v = 0.0;
            T volume = cells.volume(cell_i);
            for (size_t face_idx = 0; face_idx < cell_face_ids.size(); face_idx++) {
                size_t i_face = cell_face_ids(face_idx);
                T area = interfaces.area(i_face);
                T dot = vx * interfaces.norm().x(i_face) +
                        vy * interfaces.norm().y(i_face) +
                        vz * interfaces.norm().z(i_face);
                T sig_vel =
                    Ibis::abs(dot) + gas_model.speed_of_sound(flow_state.gas, cell_i);
                spectral_radii_c += sig_vel * area;

                if (viscous) {
                    T gamma = gas_model.gamma();
                    T mu = trans_prop.viscosity(face_fs.gas, gas_model, i_face);
                    T k = trans_prop.thermal_conductivity(face_fs.gas, gas_model, i_face);
                    T rho = face_fs.gas.rho(i_face);
                    T Pr = mu * gas_model.Cp() / k;
                    T tmp = (gamma / rho) * (mu / Pr) * area * area;
                    spectral_radii_v += tmp / volume;
                }
            }
            T local_dt =
                volume / (spectral_radii_c + viscous_signal_factor * spectral_radii_v);
            dt_utd = Ibis::min(Ibis::real_part(local_dt), dt_utd);
        });
}

template <typename T, class MemModel>
void FiniteVolume<T, MemModel>::apply_pre_reconstruction_bc(
    FlowStates<T>& fs, const GridBlock<MemModel, T>& grid, const IdealGas<T>& gas_model,
    const TransportProperties<T>& trans_prop) {
    for (size_t i = 0; i < bcs_.size(); i++) {
        std::shared_ptr<BoundaryCondition<T, MemModel>> bc = bcs_[i];
        Field<size_t> bc_faces = bc_interfaces_[i];
        bc->apply_pre_reconstruction(fs, grid, bc_faces, gas_model, trans_prop);
    }
}

template <typename T, class MemModel>
void FiniteVolume<T, MemModel>::apply_post_convective_flux_bc(
    const FlowStates<T>& fs, const GridBlock<MemModel, T>& grid,
    const IdealGas<T>& gas_model, const TransportProperties<T>& trans_prop) {
    for (size_t i = 0; i < bcs_.size(); i++) {
        std::shared_ptr<BoundaryCondition<T, MemModel>> bc = bcs_[i];
        Field<size_t> bc_faces = bc_interfaces_[i];
        bc->apply_post_convective_flux_actions(flux_, fs, grid, bc_faces, gas_model,
                                               trans_prop);
    }
}

template <typename T, class MemModel>
void FiniteVolume<T, MemModel>::apply_pre_viscous_grad_bc(
    FlowStates<T>& fs, const GridBlock<MemModel, T>& grid, const IdealGas<T>& gas_model,
    const TransportProperties<T>& trans_prop) {
    for (size_t i = 0; i < bcs_.size(); i++) {
        std::shared_ptr<BoundaryCondition<T, MemModel>> bc = bcs_[i];
        Field<size_t> bc_faces = bc_interfaces_[i];
        bc->apply_pre_viscous_grad(fs, grid, bc_faces, gas_model, trans_prop);
    }
}

template <typename T, class MemModel>
void FiniteVolume<T, MemModel>::flux_surface_integral(const GridBlock<MemModel, T>& grid,
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
                T area = -faces.area(face_id) * cell_faces.outsigns(cell_i)(face_i);
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

template <typename T, class MemModel>
size_t FiniteVolume<T, MemModel>::count_bad_cells(const FlowStates<T>& fs,
                                                  const size_t num_cells) {
    size_t n_bad_cells = 0;
    Kokkos::parallel_reduce(
        "FiniteVolume::count_bad_cells", num_cells,
        KOKKOS_LAMBDA(const int cell_i, size_t& n_bad_cells_utd) {
            if (fs.gas.temp(cell_i) < 0.0 || fs.gas.rho(cell_i) < 0.0 ||
                Ibis::isnan(fs.gas.rho(cell_i)) || Ibis::isinf(fs.gas.rho(cell_i))) {
                n_bad_cells_utd += 1;
            }
        },
        n_bad_cells);
    return n_bad_cells;
}

template <typename T, class MemModel>
void FiniteVolume<T, MemModel>::compute_viscous_gradient(
    FlowStates<T>& fs, const GridBlock<MemModel, T>& grid, const IdealGas<T>& gas_model,
    const TransportProperties<T>& trans_prop) {
    apply_pre_reconstruction_bc(fs, grid, gas_model, trans_prop);
    apply_pre_viscous_grad_bc(fs, grid, gas_model, trans_prop);
    viscous_flux_.compute_viscous_gradient(fs, grid, cell_grad_, grid.grad_calc());
}

template <typename T, class MemModel>
void FiniteVolume<T, MemModel>::compute_convective_gradient(
    FlowStates<T>& fs, const GridBlock<MemModel, T>& grid, const IdealGas<T>& gas_model,
    const TransportProperties<T>& trans_prop) {
    apply_pre_reconstruction_bc(fs, grid, gas_model, trans_prop);
    convective_flux_.compute_convective_gradient(fs, grid, cell_grad_, grid.grad_calc());
}

template class FiniteVolume<Ibis::real, SharedMem>;
template class FiniteVolume<Ibis::real, Mpi>;
template class FiniteVolume<Ibis::dual, SharedMem>;
template class FiniteVolume<Ibis::dual, Mpi>;
