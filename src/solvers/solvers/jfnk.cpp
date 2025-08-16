#include <finite_volume/primative_conserved_conversion.h>
#include <linear_algebra/gmres.h>
#include <solvers/jfnk.h>

template <class MemModel>
Jfnk<MemModel>::Jfnk(std::shared_ptr<PseudoTransientLinearSystem> system,
           std::unique_ptr<CflSchedule>&& cfl,
           std::shared_ptr<ConservedQuantities<Ibis::dual>> residuals, json config) {
    max_steps_ = config.at("max_steps");
    tolerance_ = config.at("tolerance");

    system_ = system;
    std::shared_ptr<LinearSystem> preconditioner = system_->preconditioner();
    preconditioner_ =
        std::dynamic_pointer_cast<PseudoTransientLinearSystem>(preconditioner);
    gmres_ = make_linear_solver(system, preconditioner_, config.at("linear_solver"));

    cfl_ = std::move(cfl);
    residual_based_cfl_ = cfl_->residual_based();
    dU_ = Ibis::Vector<Ibis::real>{"dU", system_->num_vars()};
    residuals_ = residuals;
}

template <class MemModel>
int Jfnk<MemModel>::initialise() {
    system_->eval_rhs();
    residual_norms_ = residuals_->L2_norms();
    initial_residual_norms_ = residual_norms_;
    return 0;
}

template <class MemModel>
void Jfnk<MemModel>::set_pseudo_time_step_size(Ibis::real dt_star) {
    system_->set_pseudo_time_step(dt_star);
    if (preconditioner_) {
        preconditioner_->set_pseudo_time_step(dt_star);
    }
}

template <class MemModel>
LinearSolveResult Jfnk<MemModel>::step(std::shared_ptr<Sim<Ibis::dual, MemModel>>& sim,
                             ConservedQuantities<Ibis::dual>& cq,
                             FlowStates<Ibis::dual>& fs, size_t step) {
    // dU is the change in the solution for the step,
    // our initial guess for it is zero
    dU_.zero();

    // set the time step
    stable_dt_ = sim->fv.estimate_dt(fs, sim->grid, sim->gas_model, sim->trans_prop);
    Ibis::real cfl = calculate_cfl(step);
    set_pseudo_time_step_size(cfl * stable_dt_);

    // solve the linear system of equations
    last_gmres_result_ = gmres_->solve(dU_);

    // apply the update and calculate the new residuals
    // so we can check non-linear convergence.
    // These residuals get re-used for the next step if we haven't converged.
    apply_update_(sim, cq, fs);
    system_->eval_rhs();
    residual_norms_ = residuals_->L2_norms();
    return last_gmres_result_;
}

template <class MemModel>
void Jfnk<MemModel>::apply_update_(std::shared_ptr<Sim<Ibis::dual, MemModel>>& sim,
                         ConservedQuantities<Ibis::dual>& cq,
                         FlowStates<Ibis::dual>& fs) {
    auto dU = dU_;
    size_t n_cells = sim->grid.num_cells();
    size_t n_cons = cq.n_conserved();
    Kokkos::parallel_for(
        "Jfnk::apply_update", n_cells, KOKKOS_LAMBDA(const size_t cell_i) {
            const size_t vector_idx = cell_i * n_cons;
            for (size_t cons_i = 0; cons_i < n_cons; cons_i++) {
                cq(cell_i, cons_i).real() += dU(vector_idx + cons_i);
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
                    vertex_pos(vertex_i, dim_i).real() += dU(vector_idx + dim_i);
                    vertex_pos(vertex_i, dim_i).dual() = 0.0;
                }
            });
        sim->grid.compute_geometric_data();
    }

    conserved_to_primatives(cq, fs, sim->gas_model);
}

template class Jfnk<SharedMem>;
template class Jfnk<Mpi>;
