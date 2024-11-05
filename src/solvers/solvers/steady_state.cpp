#include <finite_volume/conserved_quantities.h>
#include <finite_volume/finite_volume.h>
#include <finite_volume/primative_conserved_conversion.h>
#include <gas/flow_state.h>
#include <gas/transport_properties.h>
#include <simulation/simulation.h>
#include <solvers/cfl.h>
#include <solvers/steady_state.h>
#include <solvers/transient_linear_system.h>

#include "finite_volume/grid_motion_driver.h"

SteadyStateLinearisation::SteadyStateLinearisation(
    std::shared_ptr<Sim<Ibis::dual>> sim,
    std::shared_ptr<ConservedQuantities<Ibis::dual>> residuals,
    std::shared_ptr<ConservedQuantities<Ibis::dual>> cq,
    std::shared_ptr<FlowStates<Ibis::dual>> fs,
    std::shared_ptr<Vector3s<Ibis::dual>> vertex_vel, bool allow_reconstruction) {
    sim_ = sim;
    cq_ = cq;
    fs_ = fs;
    allow_reconstruction_ = allow_reconstruction;
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

    if (sim_->grid.moving()) {
        vertex_pos_tmp_ = Vector3s<Ibis::dual>{"SteadyStateLinearisation::vertex_vel",
                                               sim_->grid.num_vertices()};
    }
}

std::unique_ptr<LinearSystem> SteadyStateLinearisation::preconditioner() {
    return std::unique_ptr<LinearSystem>(
        new SteadyStateLinearisation(sim_, residuals_, cq_, fs_, vertex_vel_, false));
}

void SteadyStateLinearisation::matrix_vector_product(Ibis::Vector<Ibis::real>& vec,
                                                     Ibis::Vector<Ibis::real>& result) {
    // set the dual components of the conserved quantities
    size_t n_cons = n_cons_;
    auto residuals = *residuals_;
    Ibis::real dt_star = dt_star_;
    auto cq_tmp = cq_tmp_;
    auto cq = *cq_;
    Kokkos::parallel_for(
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
        Kokkos::parallel_for(
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
                              sim_->gas_model, sim_->trans_prop, allow_reconstruction_);
    } else {
        sim_->fv.compute_dudt(fs_tmp_, sim_->grid, residuals, sim_->gas_model,
                              sim_->trans_prop, allow_reconstruction_);
    }

    // set the components of vec to the dual component of dudt
    Kokkos::parallel_for(
        "SteadyStateLinearisation::set_vector", n_cells_,
        KOKKOS_LAMBDA(const size_t cell_i) {
            const size_t vector_idx = cell_i * n_cons;
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
        Kokkos::parallel_for(
            "SteadyStateLinearisation::set_vector::grid", num_vertices,
            KOKKOS_LAMBDA(const size_t vertex_i) {
                const size_t vector_idx = n_cells * n_cons + vertex_i * dim;
                for (size_t dim_i = 0; dim_i < dim; dim_i++) {
                    result(vector_idx + dim_i) =
                        1 / dt_star * vec(vector_idx + dim_i) -
                        Ibis::dual_part(vertex_vel(vertex_i, dim_i));
                }
            });
    }
}

void SteadyStateLinearisation::eval_rhs() {
    if (sim_->grid.moving()) {
        sim_->fv.compute_dudt(*fs_, *vertex_vel_, *cq_, sim_->grid, *residuals_,
                              sim_->gas_model, sim_->trans_prop, allow_reconstruction_);
    } else {
        sim_->fv.compute_dudt(*fs_, sim_->grid, *residuals_, sim_->gas_model,
                              sim_->trans_prop, allow_reconstruction_);
    }

    size_t n_cons = n_cons_;
    auto rhs = rhs_;
    auto residuals = *residuals_;
    Kokkos::parallel_for(
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
        Kokkos::parallel_for(
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

void SteadyStateLinearisation::set_rhs(Ibis::Vector<Ibis::real>& rhs) {
    // rhs_.deep_copy_space(rhs);
    rhs_ = rhs;
}

void SteadyStateLinearisation::set_pseudo_time_step(Ibis::real dt_star) {
    dt_star_ = dt_star;
}

SteadyState::SteadyState(json config, GridBlock<Ibis::dual> grid, std::string grid_dir,
                         std::string flow_dir)
    : Solver(grid_dir, flow_dir) {
    json solver_config = config.at("solver");
    sim_ = std::shared_ptr<Sim<Ibis::dual>>{new Sim<Ibis::dual>(grid, config)};

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
        json grid_config = config.at("grid");
        json grid_motion_config = grid_config.at("motion");
        auto grid_driver =
            build_grid_motion_driver<Ibis::dual>(sim_->grid, grid_motion_config);
        sim_->grid.set_motion_driver(grid_driver);
        vertex_vel_ = std::shared_ptr<Vector3s<Ibis::dual>>(
            new Vector3s<Ibis::dual>(grid.num_vertices()));
    }

    // set up the linear system and non-linear solver
    auto cfl = make_cfl_schedule(solver_config.at("cfl"));
    std::unique_ptr<PseudoTransientLinearSystem> system =
        std::unique_ptr<PseudoTransientLinearSystem>(
            new SteadyStateLinearisation(sim_, residuals_, cq_, fs_, vertex_vel_));
    jfnk_ = Jfnk(std::move(system), std::move(cfl), residuals_, solver_config);

    // configuration
    print_frequency_ = solver_config.at("print_frequency");
    plot_frequency_ = solver_config.at("plot_frequency");
    diagnostics_frequency_ = solver_config.at("diagnostics_frequency");

    // I/O
    io_ = FVIO<Ibis::dual>(config, 1);

    config_ = config;
}

int SteadyState::initialise() {
    // read grid and initial condition
    json meta_data{};
    json grid_config = config_.at("grid");
    int ic_result = io_.read(*fs_, sim_->grid, sim_->gas_model, sim_->trans_prop,
                             grid_config, meta_data, 0);
    int conversion_result = primatives_to_conserved(*cq_, *fs_, sim_->gas_model);

    // initialise the JFNK solver
    int jfnk_init = jfnk_.initialise();

    // start the diagnostics files
    if (diagnostics_frequency_ > 0) {
        // absolute residuals
        std::ofstream abs_residual_file("log/absolute_residuals.dat", std::ios_base::out);
        abs_residual_file << "step step wall_clock global mass momentum_x momentum_y "
                             "momentum_z energy\n";

        // relative residuals
        std::ofstream rel_residual_file("log/relative_residuals.dat", std::ios_base::out);
        rel_residual_file << "step step wall_clock global mass momentum_x momentum_y "
                             "momentum_z energy\n";

        write_residuals(0, 0.0);

        // gmres diagnostics
        std::ofstream gmres_diagnostics("log/gmres_diagnostics.dat", std::ios_base::out);
        gmres_diagnostics << "step converged residual tolerance n_iters\n";
    }

    return ic_result + conversion_result + jfnk_init;
}

int SteadyState::finalise() { return 0; }

int SteadyState::take_step(size_t step) {
    jfnk_.step(sim_, *cq_, *fs_, step);
    return 0;
}

bool SteadyState::print_this_step(unsigned int step) {
    return (step != 0 && step % print_frequency_ == 0);
}

bool SteadyState::residuals_this_step(unsigned int step) {
    return ((diagnostics_frequency_ > 0) && (step != 0) &&
            (step % diagnostics_frequency_ == 0));
}

bool SteadyState::plot_this_step(unsigned int step) {
    return (step != 0 && step % plot_frequency_ == 0);
}

int SteadyState::plot_solution(unsigned int step) {
    Ibis::real t = (Ibis::real)step;
    int result =
        io_.write(*fs_, sim_->fv, sim_->grid, sim_->gas_model, sim_->trans_prop, t);
    spdlog::info("  written flow solution: step {}", step);
    return result;
}

void SteadyState::print_progress(unsigned int step, Ibis::real wc) {
    Ibis::real relative_global_residual = jfnk_.relative_residual_norms().global().real();
    Ibis::real cfl = jfnk_.calculate_cfl(step);
    spdlog::info(
        "  step: {:>8}, relative global residual {:.2e}, cfl = {:.1f}, wc = {:.1f}s",
        step, relative_global_residual, cfl, wc);
}

bool SteadyState::stop_now(unsigned int step) {
    if (step >= max_step() - 1) return true;
    if (jfnk_.relative_residual_norms().global() < jfnk_.target_residual()) return true;
    return false;
}

std::string SteadyState::stop_reason(unsigned int step) {
    if (step >= max_step() - 1) return "reached max_step";
    if (jfnk_.relative_residual_norms().global() < jfnk_.target_residual()) {
        return "reached target residual";
    }
    return "Shouldn't reach here";
}

bool SteadyState::write_residuals(unsigned int step, Ibis::real wc) {
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

    const LinearSolveResult& gmres_result = jfnk_.last_gmres_result();
    std::ofstream gmres_diagnostics("log/gmres_diagnostics.dat", std::ios_base::app);
    gmres_diagnostics << step << " " << gmres_result.success << " "
                      << gmres_result.residual << " " << gmres_result.tol << " "
                      << gmres_result.n_iters << std::endl;
    return true;
}
