
#include <finite_volume/primative_conserved_conversion.h>
#include <solvers/runge_kutta.h>
#include <solvers/solver.h>
#include <spdlog/spdlog.h>

#include <limits>

#include "finite_volume/conserved_quantities.h"
#include "gas/transport_properties.h"
#include "solvers/cfl.h"

// Implementation of Butcher tableau
double ButcherTableau::a(size_t i, size_t j) { return a_[i - 1][j]; }
double ButcherTableau::b(size_t i) { return b_[i]; }
double ButcherTableau::c(size_t i) { return c_[i - 1]; }
size_t ButcherTableau::num_stages() { return num_stages_; }

RungeKutta::RungeKutta(json config, GridBlock<double>& grid, std::string grid_dir,
                       std::string flow_dir)
    : Solver(grid_dir, flow_dir) {
    // configuration
    json solver_config = config.at("solver");
    max_time_ = solver_config.at("max_time");
    max_step_ = solver_config.at("max_step");
    print_frequency_ = solver_config.at("print_frequency");
    plot_frequency_ = solver_config.at("plot_frequency");
    plot_every_n_steps_ = solver_config.at("plot_every_n_steps");
    cfl_ = make_cfl_schedule(solver_config.at("cfl"));
    dt_init_ = solver_config.at("dt_init");

    // Butcher tableau
    tableau_ = ButcherTableau(solver_config.at("butcher_tableau"));

    // gas_model
    gas_model_ = IdealGas<double>(config.at("gas_model"));
    trans_prop_ = TransportProperties<double>(config.at("transport_properties"));

    // memory
    grid_ = grid;
    int number_cells = grid_.num_total_cells();
    int dim = grid_.dim();
    flow_ = FlowStates<double>(number_cells);
    conserved_quantities_ = ConservedQuantities<double>(number_cells, dim);
    k_ = std::vector<ConservedQuantities<double>>(
        tableau_.num_stages(), ConservedQuantities<double>(number_cells, dim));
    if (tableau_.num_stages() > 1) {
        k_tmp_ = ConservedQuantities<double>(number_cells, dim);
        flow_tmp_ = FlowStates<double>(number_cells);
    }
    fv_ = FiniteVolume<double>(grid_, config);

    // progress
    time_since_last_plot_ = 0.0;
    t_ = 0.0;

    // input/output
    io_ = FVIO<double>(1);
}

int RungeKutta::initialise() {
    json meta_data;
    int ic_result = io_.read(flow_, grid_, gas_model_, meta_data, 0);
    int conversion_result =
        primatives_to_conserved(conserved_quantities_, flow_, gas_model_);
    dt_ = (dt_init_ > 0) ? dt_init_ : std::numeric_limits<double>::max();
    return ic_result + conversion_result;
}

int RungeKutta::finalise() { return 0; }

void RungeKutta::estimate_dt() {
    // choose the size of time step to take. We take the smallest of
    //   1. The stable timestep
    //   2. 1.5 x the previous time step
    //   3. The time till the next plot needs to be written
    stable_dt_ = fv_.estimate_dt(flow_, grid_, gas_model_, trans_prop_);
    double dt_startup = Kokkos::min(cfl_->eval(t_) * stable_dt_, 1.5 * dt_);
    dt_ = Kokkos::min(dt_startup, max_time_ - t_);
    if (plot_frequency_ > 0.0 && time_since_last_plot_ < plot_frequency_) {
        dt_ = Kokkos::min(dt_, plot_frequency_ - time_since_last_plot_);
    }
}

int RungeKutta::take_step() {
    // this has to be done before the estimation of dt, as may set
    // values used to estimate the stable time step. It also serves
    // as the first stage of all the runge-kutta schemes
    fv_.compute_dudt(flow_, grid_, k_[0], gas_model_, trans_prop_);

    // estimate the stable time step we can take. After this call,
    // dt_ will be set to the stable time step.
    estimate_dt();

    // the main part of the runge-kutta method. We've already done stage 0
    // so we start the loop at stage 1.
    for (size_t i = 1; i < tableau_.num_stages(); i++) {
        // The first evaluation for each row of the tabluea includes the initial state
        // so we treat it separately. Even if the coefficient for this stage is zero,
        // we do this step to make sure k_tmp_ is set correctly.
        apply_time_derivative(conserved_quantities_, k_tmp_, k_[0],
                              tableau_.a(i, 0) * dt_);

        // The remaining coefficients for this row
        for (size_t j = 1; j < i; j++) {
            // If the coefficient for this particular stage is zero,
            // then we skip to save adding zero to every number
            if (tableau_.a(i, j) < 1e-14) continue;

            // accumulate the intermediate state for the next function evaluation
            k_tmp_.apply_time_derivative(k_[j], tableau_.a(i, j) * dt_);
        }

        // The function evaluation
        conserved_to_primatives(k_tmp_, flow_tmp_, gas_model_);
        fv_.compute_dudt(flow_tmp_, grid_, k_[i], gas_model_, trans_prop_);
    }

    // Update the flow state
    for (size_t i = 0; i < tableau_.num_stages(); i++) {
        conserved_quantities_.apply_time_derivative(k_[i], tableau_.b(i) * dt_);
    }

    conserved_to_primatives(conserved_quantities_, flow_, gas_model_);

    // book keeping
    t_ += dt_;
    time_since_last_plot_ += dt_;
    return 0;
}

bool RungeKutta::print_this_step(unsigned int step) {
    if (step != 0 && step % print_frequency_ == 0) return true;
    return false;
}

bool RungeKutta::plot_this_step(unsigned int step) {
    if (plot_every_n_steps_ > 0 && step != 0 && step % plot_every_n_steps_ == 0)
        return true;

    if (plot_frequency_ > 0 && time_since_last_plot_ >= plot_frequency_ - 1e-15)
        return true;
    return false;
}

int RungeKutta::plot_solution(unsigned int step) {
    int result = io_.write(flow_, fv_, grid_, gas_model_, t_);
    time_since_last_plot_ = 0.0;
    spdlog::info("  written flow solution: step {}, time {:.6e}", step, t_);
    return result;
}

void RungeKutta::print_progress(unsigned int step, double wc) {
    spdlog::info(
        "  step: {:>8}, t = {:.6e} ({:.1f}%), dt = {:.6e} (cfl={:.1f}), wc = "
        "{:.1f}s",
        step, t_, t_ / max_time_ * 100, dt_, dt_ / stable_dt_, wc);
}
std::string RungeKutta::stop_reason(unsigned int step) {
    if (t_ >= max_time_ - 1e-15)
        return "reached max time";  // subtract small amount to avoid round off
                                    // error
    if (step >= max_step_ - 1) return "reached max step";
    return "Shouldn't reach here";
}
bool RungeKutta::stop_now(unsigned int step) {
    if (step >= max_step_ - 1) return true;
    if (t_ >= max_time_) return true;
    return false;
}
