#include <doctest/doctest.h>
#include <finite_volume/finite_volume.h>
#include <finite_volume/grid_motion_driver.h>
#include <finite_volume/primative_conserved_conversion.h>
#include <gas/flow_state.h>
#include <gas/transport_properties.h>
#include <parallel/parallel.h>
#include <simulation/simulation.h>
#include <solvers/cfl.h>
#include <solvers/steady_state.h>
#include <solvers/transient_linear_system.h>

#include <fstream>

#include "solvers/high_order_blending.h"

#ifdef Ibis_ENABLE_MPI
#include <ibis_mpi/ibis_mpi_dual.h>
#endif

template <class MemModel>
SteadyStateLinearisation<MemModel>::SteadyStateLinearisation(
    std::shared_ptr<Sim<Ibis::dual, MemModel>> sim,
    std::shared_ptr<ConservedQuantities<Ibis::dual>> residuals,
    std::shared_ptr<ConservedQuantities<Ibis::dual>> cq,
    std::shared_ptr<FlowStates<Ibis::dual>> fs,
    std::shared_ptr<Vector3s<Ibis::dual>> vertex_vel, bool local_time_stepping,
    bool allow_reconstruction, int jacobian_stencil_size) {
    sim_ = sim;
    cq_ = cq;
    fs_ = fs;
    allow_reconstruction_ = allow_reconstruction;
    jacobian_stencil_size_ = jacobian_stencil_size;
    size_t dim = sim_->grid.dim();

    size_t num_grid_vars = 0;
    if (sim_->grid.moving()) {
        num_grid_vars = sim_->grid.num_vertices() * dim;
    }

    n_total_cells_ = sim_->grid.num_total_cells();
    n_cells_ = sim_->grid.num_cells();
    n_cons_ = cq_->n_conserved();
    n_vars_ = n_cells_ * n_cons_ + num_grid_vars;

    rhs_ = Ibis::Vector<Ibis::real>{"SteadyStateLinearisation::rhs", n_vars_};
    fs_tmp_ = FlowStates<Ibis::dual>{n_total_cells_};
    cq_tmp_ = ConservedQuantities<Ibis::dual>{n_total_cells_, dim};
    residuals_ = residuals;
    vertex_vel_ = vertex_vel;

    local_time_stepping_ = local_time_stepping;

    if (sim_->grid.moving()) {
        vertex_pos_tmp_ = Vector3s<Ibis::dual>{"SteadyStateLinearisation::vertex_vel",
                                               sim_->grid.num_vertices()};
    }
}

template <class MemModel>
std::unique_ptr<LinearSystem> SteadyStateLinearisation<MemModel>::preconditioner() {
    bool local_time_stepping = false;
    int jacobian_stencil_size = 1;
    return std::unique_ptr<LinearSystem>(new SteadyStateLinearisation<MemModel>(
        sim_, residuals_, cq_, fs_, vertex_vel_, local_time_stepping_,
        local_time_stepping, jacobian_stencil_size));
}

template <class MemModel>
void SteadyStateLinearisation<MemModel>::matrix_vector_product(
    Ibis::Vector<Ibis::real>& vec, Ibis::Vector<Ibis::real>& result) {
    // set the dual components of the conserved quantities
    size_t n_cons = n_cons_;
    auto residuals = *residuals_;
    Ibis::real global_dt_star = dt_star_;
    Ibis::Array1D<Ibis::real> local_dt_star = local_dt_star_;
    auto cq_tmp = cq_tmp_;
    auto cq = *cq_;
    bool local_time_stepping = local_time_stepping_;
    Ibis::parallel_for(
        "SteadyStateLinearisation::set_dual", n_cells_,
        KOKKOS_LAMBDA(const size_t cell_i) {
            const int vector_idx = cell_i * n_cons;
            for (size_t cons_i = 0; cons_i < n_cons; cons_i++) {
                cq_tmp(cell_i, cons_i).real() = cq(cell_i, cons_i).real();
                cq_tmp(cell_i, cons_i).dual() = vec(vector_idx + cons_i);
            }
        });

    if (sim_->grid.moving()) {
        auto vertex_pos_tmp = vertex_pos_tmp_;
        int dim = sim_->grid.dim();
        auto vertex_pos = sim_->grid.vertices().positions();
        size_t n_cells = n_cells_;
        Ibis::parallel_for(
            "SteadyStateLinearisation::set_dual_grid", sim_->grid.num_vertices(),
            KOKKOS_LAMBDA(const size_t vertex_i) {
                const size_t vector_idx = n_cells * n_cons + vertex_i * dim;
                for (int dim_i = 0; dim_i < dim; dim_i++) {
                    vertex_pos_tmp(vertex_i, dim_i).real() =
                        vertex_pos(vertex_i, dim_i).real();
                    vertex_pos_tmp(vertex_i, dim_i).dual() = vec(vector_idx + dim_i);
                }
            });

        sim_->grid.set_vertex_positions(vertex_pos_tmp);
    }

    // convert the conserved quantities to primatives, ready to evaluate the residuals
    conserved_to_primatives(cq_tmp_, fs_tmp_, sim_->gas_model);

    // evaluate the residuals
    if (sim_->grid.moving()) {
        sim_->fv.compute_dudt(fs_tmp_, *vertex_vel_, *cq_, sim_->grid, residuals,
                              sim_->gas_model, sim_->trans_prop, allow_reconstruction_,
                              global_limiter_);
    } else {
        sim_->fv.compute_dudt(fs_tmp_, sim_->grid, residuals, sim_->gas_model,
                              sim_->trans_prop, allow_reconstruction_, global_limiter_);
    }

    // set the components of vec to the dual component of dudt
    Ibis::parallel_for(
        "SteadyStateLinearisation::set_vector", n_cells_,
        KOKKOS_LAMBDA(const size_t cell_i) {
            const size_t vector_idx = cell_i * n_cons;
            Ibis::real dt_star;
            if (local_time_stepping) {
                dt_star = local_dt_star(cell_i);
            } else {
                dt_star = global_dt_star;
            }
            for (size_t cons_i = 0; cons_i < n_cons; cons_i++) {
                result(vector_idx + cons_i) = 1 / dt_star * vec(vector_idx + cons_i) -
                                              Ibis::dual_part(residuals(cell_i, cons_i));
            }
        });

    if (sim_->grid.moving()) {
        size_t num_vertices = sim_->grid.num_vertices();
        auto vertex_vel = *vertex_vel_;
        size_t n_cells = n_cells_;
        int dim = sim_->grid.dim();
        Ibis::parallel_for(
            "SteadyStateLinearisation::set_vector::grid", num_vertices,
            KOKKOS_LAMBDA(const size_t vertex_i) {
                const size_t vector_idx = n_cells * n_cons + vertex_i * dim;
                Ibis::real dt_star = global_dt_star;
                for (int dim_i = 0; dim_i < dim; dim_i++) {
                    result(vector_idx + dim_i) =
                        1 / dt_star * vec(vector_idx + dim_i) -
                        Ibis::dual_part(vertex_vel(vertex_i, dim_i));
                }
            });
    }
}

template <class MemModel>
Ibis::CrsGraph<int, int> SteadyStateLinearisation<MemModel>::compute_matrix_graph() {
    // get the graph of the grid
    auto grid = sim_->grid;
    grid.compute_graph(jacobian_stencil_size_);
    auto grid_graph = grid.graph(jacobian_stencil_size_);
    auto grid_graph_h = grid_graph.host_mirror();
    grid_graph_h.deep_copy(grid_graph);

    // compute the graph of the linear system in serial on the CPU
    size_t n_cons = n_cons_;
    std::vector<int> serial_rowmap{0};
    std::vector<int> serial_entries;
    for (size_t grid_row_idx = 0; grid_row_idx < grid_graph_h.num_rows();
         grid_row_idx++) {
        int grid_row_start_idx = grid_graph_h.row_map(grid_row_idx);
        int num_values_in_row =
            grid_graph_h.row_map(grid_row_idx + 1) - grid_row_start_idx;
        for (size_t cons_i_row = 0; cons_i_row < n_cons; cons_i_row++) {
            for (int grid_col_idx = 0; grid_col_idx < num_values_in_row; grid_col_idx++) {
                int grid_col = grid_graph_h.entries(grid_row_start_idx + grid_col_idx);
                int start_idx = grid_col * n_cons;
                for (size_t cons_i = 0; cons_i < n_cons; cons_i++) {
                    serial_entries.push_back(start_idx + cons_i);
                }
            }
            serial_rowmap.push_back(serial_entries.size());
        }
    }

    // allocate the graph on the default device
    Ibis::CrsGraph<int, int> system_graph(serial_rowmap.size() - 1,
                                          serial_entries.size());
    auto system_graph_h = system_graph.host_mirror();
    for (size_t i = 0; i < serial_rowmap.size(); i++) {
        system_graph_h.row_map(i) = serial_rowmap[i];
    }
    for (size_t i = 0; i < serial_entries.size(); i++) {
        system_graph_h.entries(i) = serial_entries[i];
    }
    system_graph.deep_copy(system_graph_h);

    // return the system graph
    return system_graph;
}

template <class MemModel>
void SteadyStateLinearisation<MemModel>::compute_matrix(
    Ibis::CrsMatrix<int, int, Ibis::real>& matrix) {
    auto grid = sim_->grid;
    size_t num_cells = grid.num_cells();
    if (pert_vec_.size() < n_vars_) {
        pert_vec_ =
            Ibis::Vector<Ibis::real>("SteadyStateLineariation::purt_vec", n_vars_);
    }
    if (res_vec_.size() < n_vars_) {
        res_vec_ = Ibis::Vector<Ibis::real>("SteadyStateLinearisation::res_vec", n_vars_);
    }
    if (grid.colours().size() == 0) {
        grid.compute_colours();
    }
    auto pert_vec = pert_vec_;
    auto res_vec = res_vec_;
    int num_colours = grid.num_colours();
    int stencil_distance = jacobian_stencil_size_;
    size_t n_cons = n_cons_;
    auto colours = grid.colours();
    auto neighbours = grid.cells().neighbour_cells();
    Kokkos::deep_copy(matrix.values, 0.0);
    for (int colour = 1; colour < num_colours + 1; colour++) {
        for (size_t perturbed_conserved_i = 0; perturbed_conserved_i < n_cons_;
             perturbed_conserved_i++) {
            // set values in the perturbation vector
            Ibis::parallel_for(
                "SteadyStateLinearisation::set_purturbation_vec", num_cells,
                KOKKOS_LAMBDA(const size_t cell_i) {
                    for (size_t cons_i = 0; cons_i < n_cons; cons_i++) {
                        size_t vector_idx = cell_i * n_cons + cons_i;
                        int cell_colour = colours(cell_i);
                        bool perturb_entry =
                            (cell_colour == colour) && (cons_i == perturbed_conserved_i);
                        pert_vec(vector_idx) = (perturb_entry) ? 1.0 : 0.0;
                    }
                });

            // Perform matrix-vector product
            matrix_vector_product(pert_vec, res_vec);

            // extract matrix elements
            Ibis::parallel_for(
                "SteadyStateLinearisation::set_crs", num_cells,
                KOKKOS_LAMBDA(const size_t cell_i) {
                    // This essentially does a depth first search of the cell's neighbours
                    // to find a cell that was perturbed. Maybe breadth first would be
                    // better?
                    if (colours(cell_i) == colour) {
                        // this cell was perturbed
                        size_t col_idx = cell_i * n_cons + perturbed_conserved_i;
                        for (size_t affected_conserved_i = 0;
                             affected_conserved_i < n_cons; affected_conserved_i++) {
                            size_t row_idx = cell_i * n_cons + affected_conserved_i;
                            matrix(row_idx, col_idx) = res_vec(row_idx);
                        }
                        return;
                    } else if (stencil_distance >= 1) {
                        // Check if the neighbours of this cell was perturbed
                        auto cell_i_ngbrs = neighbours(cell_i);
                        for (size_t ngbr_i = 0; ngbr_i < cell_i_ngbrs.size(); ngbr_i++) {
                            size_t ngbr_cell = cell_i_ngbrs(ngbr_i);
                            if (ngbr_cell < num_cells) {
                                if (colours(ngbr_cell) == colour) {
                                    // cell_i was perturbed by ngbr_cell
                                    size_t col_idx =
                                        ngbr_cell * n_cons + perturbed_conserved_i;
                                    for (size_t affected_cons_i = 0;
                                         affected_cons_i < n_cons; affected_cons_i++) {
                                        size_t row_idx =
                                            cell_i * n_cons + affected_cons_i;
                                        matrix(row_idx, col_idx) = res_vec(row_idx);
                                    }
                                    return;
                                } else if (stencil_distance >= 2) {
                                    auto ngbr_ngbrs = neighbours(ngbr_cell);
                                    for (size_t ngbr_ngbr_i = 0;
                                         ngbr_ngbr_i < ngbr_ngbrs.size(); ngbr_ngbr_i++) {
                                        size_t ngbr_ngbr_cell = ngbr_ngbrs(ngbr_ngbr_i);
                                        if (ngbr_ngbr_cell < num_cells &&
                                            ngbr_ngbr_cell != cell_i) {
                                            if (colours(ngbr_ngbr_cell) == colour) {
                                                size_t col_idx = ngbr_ngbr_cell * n_cons +
                                                                 perturbed_conserved_i;
                                                for (size_t affected_cons_i = 0;
                                                     affected_cons_i < n_cons;
                                                     affected_cons_i++) {
                                                    size_t row_idx =
                                                        cell_i * n_cons + affected_cons_i;
                                                    matrix(row_idx, col_idx) =
                                                        res_vec(row_idx);
                                                }
                                                return;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                });
        }
    }
}

template <class MemModel>
void SteadyStateLinearisation<MemModel>::eval_rhs() {
    if (sim_->grid.moving()) {
        sim_->fv.compute_dudt(*fs_, *vertex_vel_, *cq_, sim_->grid, *residuals_,
                              sim_->gas_model, sim_->trans_prop, allow_reconstruction_,
                              global_limiter_);
    } else {
        sim_->fv.compute_dudt(*fs_, sim_->grid, *residuals_, sim_->gas_model,
                              sim_->trans_prop, allow_reconstruction_, global_limiter_);
    }

    size_t n_cons = n_cons_;
    auto rhs = rhs_;
    auto residuals = *residuals_;
    Ibis::parallel_for(
        "SteadyStateLinearisation::eval_rhs", n_cells_, KOKKOS_LAMBDA(const int cell_i) {
            const size_t vector_idx = cell_i * n_cons;
            for (size_t cons_i = 0; cons_i < n_cons; cons_i++) {
                rhs(vector_idx + cons_i) = Ibis::real_part(residuals(cell_i, cons_i));
            }
        });

    if (sim_->grid.moving()) {
        size_t n_cells = n_cells_;
        size_t num_vertices = sim_->grid.num_vertices();
        int dim = sim_->grid.dim();
        auto vertex_vel = *vertex_vel_;
        Ibis::parallel_for(
            "SteadyStateLinearisation::eval_rhs::grid", num_vertices,
            KOKKOS_LAMBDA(const size_t vertex_i) {
                const size_t vector_idx = n_cells * n_cons + vertex_i * dim;
                for (int dim_i = 0; dim_i < dim; dim_i++) {
                    rhs(vector_idx + dim_i) =
                        Ibis::real_part(vertex_vel(vertex_i, dim_i));
                }
            });
    }
}

template <class MemModel>
void SteadyStateLinearisation<MemModel>::set_rhs(Ibis::Vector<Ibis::real>& rhs) {
    // rhs_.deep_copy_space(rhs);
    rhs_ = rhs;
}

template <class MemModel>
void SteadyStateLinearisation<MemModel>::set_pseudo_time_step(Ibis::real dt_star) {
    dt_star_ = dt_star;
}

template <class MemModel>
void SteadyStateLinearisation<MemModel>::set_local_pseudo_time_step(
    Ibis::Array1D<Ibis::real>& local_dt_star) {
    local_dt_star_ = local_dt_star;
}

template <class MemModel>
void SteadyStateLinearisation<MemModel>::set_global_limiter(Ibis::real global_limiter) {
    global_limiter_ = global_limiter;
}

template class SteadyStateLinearisation<SharedMem>;
template class SteadyStateLinearisation<Mpi>;

template <class MemModel>
SteadyState<MemModel>::SteadyState(json config, GridBlock<MemModel, Ibis::dual> grid,
                                   std::string grid_dir, std::string flow_dir)
    : Solver(grid_dir, flow_dir) {
    json solver_config = config.at("solver");
    sim_ = std::shared_ptr<Sim<Ibis::dual, MemModel>>{
        new Sim<Ibis::dual, MemModel>(grid, config)};

    size_t n_total_cells = sim_->grid.num_total_cells();
    // size_t n_cells = sim_->grid.num_cells();
    size_t dim = sim_->grid.dim();

    fs_ = std::shared_ptr<FlowStates<Ibis::dual>>{
        new FlowStates<Ibis::dual>(n_total_cells)};

    cq_ = std::shared_ptr<ConservedQuantities<Ibis::dual>>{
        new ConservedQuantities<Ibis::dual>(n_total_cells, dim)};

    residuals_ = std::shared_ptr<ConservedQuantities<Ibis::dual>>{
        new ConservedQuantities<Ibis::dual>(n_total_cells, dim)};

    if (sim_->grid.moving()) {
        json grid_config = config.at("grids")[grid.id()];
        json grid_motion_config = grid_config.at("motion");
        auto grid_driver =
            build_grid_motion_driver<Ibis::dual>(sim_->grid, grid_motion_config);
        sim_->grid.set_motion_driver(grid_driver);
        vertex_vel_ = std::shared_ptr<Vector3s<Ibis::dual>>(
            new Vector3s<Ibis::dual>(grid.num_vertices()));
    }

    bool local_time_stepping_ = solver_config.at("local_time_stepping");
    int reconstruction_order =
        config.at("convective_flux").at("reconstruction_order").at("order");
    int allow_reconstruction = reconstruction_order > 1.0;

    // set up the linear system and non-linear solver
    std::unique_ptr<PseudoTransientLinearSystem> system =
        std::unique_ptr<PseudoTransientLinearSystem>(
            new SteadyStateLinearisation<MemModel>(sim_, residuals_, cq_, fs_,
                                                   vertex_vel_, local_time_stepping_,
                                                   allow_reconstruction, 1));

    auto cfl = make_cfl_schedule(solver_config.at("cfl"));
    auto high_order_blending = make_high_order_blending_schedule(
        config.at("convective_flux").at("reconstruction_order"));
    jfnk_ = Jfnk<MemModel>(std::move(system), std::move(cfl),
                           std::move(high_order_blending), residuals_, solver_config);

    if (local_time_stepping_) {
        local_pseudo_time_step_ =
            Ibis::Array1D<Ibis::real>{"local_dt", sim_->grid.num_cells()};
        jfnk_.set_local_pseudo_time_step_size(local_pseudo_time_step_);
    }

    // configuration
    print_frequency_ = solver_config.at("print_frequency");
    plot_frequency_ = solver_config.at("plot_frequency");
    diagnostics_frequency_ = solver_config.at("diagnostics_frequency");

    // I/O
    io_ = FVIO<Ibis::dual, MemModel>(config, 1);

    config_ = config;
}

template <class MemModel>
int SteadyState<MemModel>::initialise() {
    // read grid and initial condition
    json meta_data{};
    json grid_config = config_.at("grids")[sim_->grid.id()];
    int ic_result = io_.read(*fs_, sim_->grid, sim_->gas_model, sim_->trans_prop,
                             grid_config, meta_data, 0);
    int conversion_result = primatives_to_conserved(*cq_, *fs_, sim_->gas_model);

    // initialise the JFNK solver
    int jfnk_init = jfnk_.initialise();

    // start the diagnostics files
    if (diagnostics_frequency_ > 0) {
        // absolute residuals
        std::ofstream abs_residual_file("log/absolute_residuals.dat", std::ios_base::out);
        abs_residual_file << "step sim_time wall_clock global mass momentum_x momentum_y "
                             "momentum_z energy\n";

        // relative residuals
        std::ofstream rel_residual_file("log/relative_residuals.dat", std::ios_base::out);
        rel_residual_file << "step sim_time wall_clock global mass momentum_x momentum_y "
                             "momentum_z energy\n";

        write_residuals(0, 0.0);

        // solver diagnostics
        std::ofstream gmres_diagnostics("log/solver_diagnostics.dat", std::ios_base::out);
        gmres_diagnostics
            << "step converged linear_residual tolerance n_iters relaxation_factor cfl\n";
    }

    return ic_result + conversion_result + jfnk_init;
}

template <class MemModel>
int SteadyState<MemModel>::finalise() {
    return 0;
}

template <class MemModel>
bool SteadyState<MemModel>::is_master_() {
    return (Ibis::get_world_rank<MemModel>() == 0);
}

template <class MemModel>
int SteadyState<MemModel>::take_step(size_t step) {
    jfnk_.step(sim_, *cq_, *fs_, step);
    return 0;
}

template <class MemModel>
bool SteadyState<MemModel>::print_this_step(unsigned int step) {
    return (step != 0 && step % print_frequency_ == 0);
}

template <class MemModel>
bool SteadyState<MemModel>::residuals_this_step(unsigned int step) {
    return ((diagnostics_frequency_ > 0) && (step != 0) &&
            (step % diagnostics_frequency_ == 0));
}

template <class MemModel>
bool SteadyState<MemModel>::plot_this_step(unsigned int step) {
    return (step != 0 && step % plot_frequency_ == 0);
}

template <class MemModel>
int SteadyState<MemModel>::plot_solution(unsigned int step) {
    Ibis::real t = (Ibis::real)step;
    int result =
        io_.write(*fs_, sim_->fv, sim_->grid, sim_->gas_model, sim_->trans_prop, t);
    io_.increment_time_index();
    spdlog::info("  written flow solution: step {}", step);
    return result;
}

template <class MemModel>
void SteadyState<MemModel>::print_progress(unsigned int step, Ibis::real wc) {
    Ibis::real relative_global_residual = jfnk_.relative_residual_norms().global().real();
    Ibis::real cfl = jfnk_.cfl();
    spdlog::info(
        "  step: {:>8}, relative global residual {:.2e}, cfl = {:.1f}, wc = {:.1f}s",
        step, relative_global_residual, cfl, wc);
}

template <class MemModel>
bool SteadyState<MemModel>::stop_now(unsigned int step) {
    if (step >= max_step() - 1) return true;
    if (jfnk_.relative_residual_norms().global() < jfnk_.target_residual()) return true;
    return false;
}

template <class MemModel>
std::string SteadyState<MemModel>::stop_reason(unsigned int step) {
    if (step >= max_step() - 1) return "reached max_step";
    if (jfnk_.relative_residual_norms().global() < jfnk_.target_residual()) {
        return "reached target residual";
    }
    return "Shouldn't reach here";
}

template <class MemModel>
bool SteadyState<MemModel>::write_residuals(unsigned int step, Ibis::real wc) {
    spdlog::debug("Writing residuals at step {}", step);

    // the absolute residuals
    ConservedQuantitiesNorm<Ibis::dual> abs_norms = jfnk_.residual_norms();
    std::ofstream residual_file("log/absolute_residuals.dat", std::ios_base::app);
    abs_norms.write_to_file(residual_file, wc, (Ibis::real)step, step);

    // the relative residuals
    ConservedQuantitiesNorm<Ibis::dual> rel_norms = jfnk_.relative_residual_norms();
    std::ofstream relative_residual_file("log/relative_residuals.dat",
                                         std::ios_base::app);
    rel_norms.write_to_file(relative_residual_file, wc, (Ibis::real)step, step);

    const typename Jfnk<MemModel>::StepResult& step_result = jfnk_.last_step_result();
    Ibis::real cfl = jfnk_.cfl();
    std::ofstream gmres_diagnostics("log/solver_diagnostics.dat", std::ios_base::app);
    gmres_diagnostics << step << " " << step_result.linear_solver_result.success << " "
                      << step_result.linear_solver_result.residual << " "
                      << step_result.linear_solver_result.tol << " "
                      << step_result.linear_solver_result.n_iters << " "
                      << step_result.relaxation_factor << " " << cfl << std::endl;
    return true;
}

template class SteadyState<SharedMem>;
template class SteadyState<Mpi>;

TEST_CASE("steady_state coloured jacobian") {
    std::ifstream f("../../../src/solvers/test/config.json");
    json config = json::parse(f);
    config["grid_file_name"] = "grid.su2";
    GridBlock<SharedMem, Ibis::dual> grid("../../../src/solvers/test",
                                          config.at("grids")[0]);
    auto sim = std::make_shared<Sim<Ibis::dual, SharedMem>>(grid, config);
    auto residuals =
        std::make_shared<ConservedQuantities<Ibis::dual>>(grid.num_cells(), grid.dim());
    auto cq = std::make_shared<ConservedQuantities<Ibis::dual>>(grid.num_total_cells(),
                                                                grid.dim());
    auto fs = std::make_shared<FlowStates<Ibis::dual>>(grid.num_total_cells());

    // set flow states
    for (size_t cell_i = 0; cell_i < grid.num_total_cells(); cell_i++) {
        fs->gas.rho(cell_i) = 0.01;
        fs->gas.temp(cell_i) = 300;
        fs->vel.x(cell_i) = 100.0;
        fs->vel.y(cell_i) = 0.0;
        fs->vel.z(cell_i) = 0.0;
    }
    sim->gas_model.update_thermo_from_rhoT(fs->gas);
    primatives_to_conserved(*cq, *fs, sim->gas_model);

    // set up the linear system
    SteadyStateLinearisation<SharedMem> system(sim, residuals, cq, fs, nullptr, false,
                                               true, 2);
    SteadyStateLinearisation<SharedMem> system_alt(sim, residuals, cq, fs, nullptr, false,
                                                   true, 2);
    system.set_pseudo_time_step(1);
    system_alt.set_pseudo_time_step(1);

    // compute the matrix
    auto matrix_graph = system.compute_matrix_graph();
    Ibis::CrsMatrix<int, int, Ibis::real> matrix(matrix_graph);
    system.compute_matrix(matrix);

    // check the entries one by perturbing one primative at a time
    size_t n_cons = grid.dim() + 2;
    size_t n_vars = n_cons * grid.num_cells();
    Ibis::Vector<Ibis::real> pert_vec = Ibis::Vector<Ibis::real>("purt_vec", n_vars);
    Ibis::Vector<Ibis::real> res_vec = Ibis::Vector<Ibis::real>("purt_vec", n_vars);
    for (size_t cell_i = 0; cell_i < grid.num_cells(); cell_i++) {
        for (size_t perturbed_cons_i = 0; perturbed_cons_i < n_cons; perturbed_cons_i++) {
            // compute contributions of perturbing one variable at a time
            size_t vector_idx = cell_i * n_cons + perturbed_cons_i;
            res_vec.zero();
            pert_vec.zero();
            pert_vec(vector_idx) = 1.0;
            system_alt.matrix_vector_product(pert_vec, res_vec);

            // check the column of the jacobian matrix against the single column
            size_t col_i = cell_i * n_cons + perturbed_cons_i;
            for (size_t row_i = 0; row_i < n_vars; row_i++) {
                size_t affected_cons_i = row_i % n_cons;
                INFO("row_i = ", row_i, " col_i = ", col_i, " cell_i = ", cell_i,
                     " perturbed_cons_i = ", perturbed_cons_i,
                     " affected_cons_i = ", affected_cons_i);
                if (matrix.graph.contains_entry(row_i, col_i)) {
                    CHECK(matrix(row_i, col_i) == doctest::Approx(res_vec(row_i)));
                } else {
                    CHECK(0 == doctest::Approx(res_vec(row_i)));
                }
            }
        }
    }
}
