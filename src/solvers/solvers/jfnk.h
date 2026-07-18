#ifndef JFNK_H
#define JFNK_H

#include <gas/flow_state.h>
#include <gas/transport_properties.h>
#include <io/io.h>
#include <linear_algebra/gmres.h>
#include <linear_algebra/linear_system.h>
#include <solvers/cfl.h>
#include <solvers/high_order_blending.h>
#include <solvers/transient_linear_system.h>
#include <util/conserved_quantities.h>
#include <util/numeric_types.h>

#include <memory>
#include <nlohmann/json.hpp>
#include "linear_algebra/linear_solver.h"

using json = nlohmann::json;

template <class MemModel>
class Jfnk {
public:
    Jfnk() {}

    Jfnk(std::shared_ptr<PseudoTransientLinearSystem> system,
         std::unique_ptr<CflSchedule>&& cfl,
         std::unique_ptr<HighOrderBlendingSchedule>&& high_order_blending,
         std::shared_ptr<ConservedQuantities<Ibis::dual>> resiudals, json config);

    int initialise();

    struct StepResult {
        LinearSolveResult linear_solver_result;
        Ibis::real relaxation_factor = 1.0;
        size_t num_bad_cells = 0;
    };

    StepResult step(std::shared_ptr<Sim<Ibis::dual, MemModel>>& sim,
                    ConservedQuantities<Ibis::dual>& cq, FlowStates<Ibis::dual>& fs,
                    size_t step_n);

    void solve(std::shared_ptr<Sim<Ibis::dual, MemModel>>& sim);

    size_t max_steps() const { return max_steps_; }

    Ibis::real pseudo_time_step_size() const { return stable_dt_; }

    Ibis::real cfl() const { return cfl_value_; }

    void update_cfl(size_t step) {
        Ibis::real cfl;
        if (auto* residual_based_cfl = dynamic_cast<ResidualBasedCfl*>(cfl_.get())) {
            // If the last step failed, reduce the CFL
            // if (last_step_result_.num_bad_cells > 0) {
            residual_based_cfl->reduce_cfl(last_step_result_.relaxation_factor);
            // }
            cfl = cfl_->eval(Ibis::real_part(relative_residual_norms().global()));
        } else {
            cfl = cfl_->eval((Ibis::real)step);
        }
        cfl_value_ = cfl;
    }

    Ibis::real calculate_global_limiter() const {
        if (auto* high_order_blending =
                dynamic_cast<LinearResidualBasedHighOrderBlending*>(
                    high_order_blending_.get())) {
            high_order_blending->set_residual(
                Ibis::real_part(relative_residual_norms().global()));
        }
        return high_order_blending_->eval_global_limiter();
    }

    ConservedQuantitiesNorm<Ibis::dual> residual_norms() const { return residual_norms_; }

    ConservedQuantitiesNorm<Ibis::dual> relative_residual_norms() const {
        return residual_norms_ / initial_residual_norms_;
    }

    Ibis::real target_residual() const { return tolerance_; }

    const StepResult& last_step_result() const { return last_step_result_; }

    void set_pseudo_time_step_size(Ibis::real dt_star);
    void set_local_pseudo_time_step_size(Ibis::Array1D<Ibis::real>& local_dt_star_);

private:
    // linear solver
    // The JFNK object owns the precondition system and solver, because it has the
    // full view of the health of the non-linear solver, and whether the preconditoner
    // needs to be updated, and it has access to the solution for the reduce basis
    // preconditioner
    std::shared_ptr<PseudoTransientLinearSystem> system_;
    std::unique_ptr<IterativeLinearSolver> gmres_;
    std::shared_ptr<PseudoTransientLinearSystem> precondition_system_;
    std::shared_ptr<DirectLinearSolver> precondition_solver_;

    std::unique_ptr<CflSchedule> cfl_;
    std::unique_ptr<HighOrderBlendingSchedule> high_order_blending_;
    Ibis::Vector<Ibis::real> dU_;

    bool local_time_stepping_;
    Ibis::real stable_dt_;
    Ibis::Array1D<Ibis::real> local_pseudo_dt_;
    Ibis::real cfl_value_;

    size_t max_steps_;
    Ibis::real tolerance_;
    size_t gmres_iters_to_recompute_preconditioner_ = 20;
    size_t preconditioner_update_interval_;

    Ibis::real min_relaxation_factor_;
    Ibis::real physicality_check_under_relaxation_factor_;
    Ibis::real cfl_reduction_factor_;

    std::shared_ptr<ConservedQuantities<Ibis::dual>> residuals_;
    ConservedQuantitiesNorm<Ibis::dual> residual_norms_;
    ConservedQuantitiesNorm<Ibis::dual> initial_residual_norms_;
    StepResult last_step_result_;
    bool residual_based_cfl_;

    void set_global_limiter(Ibis::real global_limiter);

    void init_linear_solver(json config);
    void update_preconditioner(size_t step);

public:  // this is public to appease NVCC
    void apply_update_(std::shared_ptr<Sim<Ibis::dual, MemModel>>& sim,
                       ConservedQuantities<Ibis::dual>& cq, FlowStates<Ibis::dual>& fs,
                       Ibis::real factor);
};

#endif
