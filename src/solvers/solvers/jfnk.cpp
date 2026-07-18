#include <finite_volume/primative_conserved_conversion.h>
#include <linear_algebra/gmres.h>
#include <solvers/jfnk.h>
#include <memory>

#include <linear_algebra/ilu.h>
#include <linear_algebra/reduced_basis_preconditioner.h>
#include <solvers/high_order_blending.h>

#ifdef Ibis_ENABLE_MPI
#include <ibis_mpi/ibis_mpi_conserved_quantities.h>
#include <ibis_mpi/ibis_mpi_dual.h>
#endif

template <class MemModel>
Jfnk<MemModel>::Jfnk(std::shared_ptr<PseudoTransientLinearSystem> system,
                     std::unique_ptr<CflSchedule>&& cfl,
                     std::unique_ptr<HighOrderBlendingSchedule>&& high_order_blending,
                     std::shared_ptr<ConservedQuantities<Ibis::dual>> residuals,
                     json config) {
    max_steps_ = config.at("max_steps");
    tolerance_ = config.at("tolerance");

    system_ = system;
    init_linear_solver(config);

    // Init some other settings
    local_time_stepping_ = config.at("local_time_stepping");
    min_relaxation_factor_ = config.at("min_relaxation_factor");
    physicality_check_under_relaxation_factor_ =
        config.at("physicality_check_under_relaxation_factor");
    cfl_reduction_factor_ = config.at("cfl_reduction_factor");
    preconditioner_update_interval_ = config.at("preconditioner_update_interval");

    cfl_ = std::move(cfl);
    high_order_blending_ = std::move(high_order_blending);
    residual_based_cfl_ = cfl_->residual_based();
    dU_ = Ibis::Vector<Ibis::real>{"dU", system_->num_vars()};
    residuals_ = residuals;
}

template <class MemModel>
void Jfnk<MemModel>::init_linear_solver(json config) {
    // Init the linear solver
    precondition_system_ = nullptr;
    precondition_solver_ = nullptr;
    if (config.at("linear_solver").at("type") == "gmres") {
        json preconditioner_config = config.at("linear_solver").at("preconditioner");
        if (preconditioner_config.at("type") == "ilu") {
            std::shared_ptr<LinearSystem> precondition_system = system_->preconditioner();
            precondition_system_ =
                std::dynamic_pointer_cast<PseudoTransientLinearSystem>(precondition_system);
            gmres_iters_to_recompute_preconditioner_ =
                config.at("linear_solver")
                    .at("preconditioner")
                    .at("gmres_iters_before_recompute");
            size_t fill_in = preconditioner_config.at("fill_in");
            precondition_solver_ = std::make_shared<ILU<MemModel>>(precondition_system_, fill_in);
        } else if (preconditioner_config.at("type") == "reduced_basis") {
            std::shared_ptr<LinearSystem> precondition_system = system_->preconditioner();
            precondition_system_ =
                std::dynamic_pointer_cast<PseudoTransientLinearSystem>(precondition_system);
            size_t num_bases = preconditioner_config.at("basis_rank");
            precondition_solver_ = std::make_shared<ReducedBasisPreconditioner<MemModel>>(precondition_system_, num_bases);
        } else if (preconditioner_config.at("type") == "none") {
            // nothing to do
        } else {
            std::string unknown_preconditioner = preconditioner_config.at("type");
            spdlog::error("Unknown preconditioner: {}", unknown_preconditioner);
            throw new std::runtime_error("Unknown preconditioner");
        }
        gmres_ = std::make_unique<Gmres>(system_, precondition_system_, precondition_solver_,
                                        config.at("linear_solver"));
    } else if (config.at("linear_solver").at("type") == "fgmres") {
        std::shared_ptr<LinearSystem> precondition_system = system_->preconditioner();
        precondition_system_ =
            std::dynamic_pointer_cast<PseudoTransientLinearSystem>(precondition_system);
        gmres_ = std::make_unique<FGmres>(system_, precondition_system_,
                                          config.at("linear_solver"));
    } else {
        std::string unknown_linear_solver = config.at("linear_solver").at("type");
        spdlog::error("Unknown linear solver: {}", unknown_linear_solver);
        throw new std::runtime_error("Unknown linear solver");
    }
}

template <class MemModel>
int Jfnk<MemModel>::initialise() {
    system_->eval_rhs();
    update_cfl(0);
    residual_norms_ = residuals_->L2_norms<MemModel>();
    initial_residual_norms_ = residual_norms_;

    // gmres_->update_preconditioner();
    return 0;
}

template <class MemModel>
void Jfnk<MemModel>::set_pseudo_time_step_size(Ibis::real dt_star) {
    system_->set_pseudo_time_step(dt_star);
    if (precondition_system_) {
        precondition_system_->set_pseudo_time_step(dt_star);
    }
}

template <class MemModel>
void Jfnk<MemModel>::set_local_pseudo_time_step_size(
    Ibis::Array1D<Ibis::real>& local_dt_star) {
    local_pseudo_dt_ = local_dt_star;
    system_->set_local_pseudo_time_step(local_dt_star);
    if (precondition_system_) {
        precondition_system_->set_local_pseudo_time_step(local_dt_star);
    }
}

template <class MemModel>
void Jfnk<MemModel>::set_global_limiter(Ibis::real global_limiter) {
    system_->set_global_limiter(global_limiter);
    if (precondition_system_) {
        precondition_system_->set_global_limiter(global_limiter);
    }
}

template <class MemModel>
void Jfnk<MemModel>::update_preconditioner(size_t step) {
    if (std::dynamic_pointer_cast<ILU<MemModel>>(precondition_solver_)) {
        auto ilu = std::dynamic_pointer_cast<ILU<MemModel>>(precondition_solver_);
        if ((last_step_result_.linear_solver_result.n_iters >
                gmres_iters_to_recompute_preconditioner_) ||
            (!last_step_result_.linear_solver_result.success) ||
            (step % preconditioner_update_interval_ == 0)) {
            spdlog::debug("Updating ILU decomposition at step {}", step);
            ilu->update_decomposition();
        }    
    } else if (std::dynamic_pointer_cast<ReducedBasisPreconditioner<MemModel>>(precondition_solver_)) {
        auto rb = std::dynamic_pointer_cast<ReducedBasisPreconditioner<MemModel>>(precondition_solver_);
        rb->update_basis(dU_);
    }
}

template <class MemModel>
Jfnk<MemModel>::StepResult Jfnk<MemModel>::step(
    std::shared_ptr<Sim<Ibis::dual, MemModel>>& sim, ConservedQuantities<Ibis::dual>& cq,
    FlowStates<Ibis::dual>& fs, size_t step_n) {

    // set the time step
    if (last_step_result_.linear_solver_result.success) {
        update_cfl(step_n);
    }
    if (local_time_stepping_) {
        sim->fv.estimate_dt(local_pseudo_dt_, fs, sim->grid, sim->gas_model,
                            sim->trans_prop, cfl_value_);
        set_local_pseudo_time_step_size(local_pseudo_dt_);
    } else {
        stable_dt_ = sim->fv.estimate_dt(fs, sim->grid, sim->gas_model, sim->trans_prop);
        set_pseudo_time_step_size(cfl_value_ * stable_dt_);
    }

    set_global_limiter(calculate_global_limiter());
    update_preconditioner(step_n);

    // solve the linear system of equations
    // dU is the change in the solution for the step,
    // our initial guess for it is zero
    dU_.zero();
    LinearSolveResult last_gmres_result = gmres_->solve(dU_);

    Ibis::real relaxation_factor = 1.0;
    size_t num_bad_cells = 0;
    while (relaxation_factor > min_relaxation_factor_) {
        apply_update_(sim, cq, fs, relaxation_factor);
        num_bad_cells = sim->fv.count_bad_cells(fs, sim->grid.num_cells());
        if (num_bad_cells == 0) {
            break;
        }
        apply_update_(sim, cq, fs, -relaxation_factor);
        relaxation_factor *= physicality_check_under_relaxation_factor_;
    }
    last_step_result_ = StepResult{last_gmres_result, relaxation_factor, num_bad_cells};

    // calculate the new residuals so we can check non-linear convergence.
    // These residuals get re-used for the next step if we haven't converged.
    system_->eval_rhs();
    residual_norms_ = residuals_->L2_norms<MemModel>();
    return last_step_result_;
}

template <class MemModel>
void Jfnk<MemModel>::apply_update_(std::shared_ptr<Sim<Ibis::dual, MemModel>>& sim,
                                   ConservedQuantities<Ibis::dual>& cq,
                                   FlowStates<Ibis::dual>& fs, Ibis::real factor) {
    auto dU = dU_;
    size_t n_cells = sim->grid.num_cells();
    size_t n_cons = cq.n_conserved();
    Kokkos::parallel_for(
        "Jfnk::apply_update", n_cells, KOKKOS_LAMBDA(const size_t cell_i) {
            const size_t vector_idx = cell_i * n_cons;
            for (size_t cons_i = 0; cons_i < n_cons; cons_i++) {
                cq(cell_i, cons_i).real() += factor * dU(vector_idx + cons_i);
                cq(cell_i, cons_i).dual() = 0.0;
            }
        });
    if (sim->grid.moving()) {
        size_t n_vertices = sim->grid.num_vertices();
        int dim = sim->grid.dim();
        auto vertex_pos = sim->grid.vertices().positions();
        Kokkos::parallel_for(
            "Jfnk::apply_update::grid", n_vertices, KOKKOS_LAMBDA(const size_t vertex_i) {
                const size_t vector_idx = n_cells * n_cons + vertex_i * dim;
                for (int dim_i = 0; dim_i < dim; dim_i++) {
                    vertex_pos(vertex_i, dim_i).real() += factor * dU(vector_idx + dim_i);
                    vertex_pos(vertex_i, dim_i).dual() = 0.0;
                }
            });
        sim->grid.compute_geometric_data();
    }

    conserved_to_primatives(cq, fs, sim->gas_model);
}

template class Jfnk<SharedMem>;
template class Jfnk<Mpi>;
