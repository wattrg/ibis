
#include <finite_volume/primative_conserved_conversion.h>
#include <solvers/runge_kutta.h>
#include <solvers/solver.h>
#include <spdlog/spdlog.h>
#include <limits>

#include "gas/transport_properties.h"
#include "solvers/cfl.h"

RungeKutta::RungeKutta(json config, GridBlock<double>& grid,
                       std::string grid_dir, std::string flow_dir)
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

    // gas_model
    gas_model_ = IdealGas<double>(config.at("gas_model"));
    trans_prop_ =
        TransportProperties<double>(config.at("transport_properties"));

    // memory
    grid_ = grid;
    int number_cells = grid_.num_total_cells();
    int dim = grid_.dim();
    flow_ = FlowStates<double>(number_cells);
    conserved_quantities_ = ConservedQuantities<double>(number_cells, dim);
    dUdt_ = ConservedQuantities<double>(number_cells, dim);
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

int RungeKutta::take_step() {
    // this has to be done before the estimation of dt, as may set
    // values used to estimate the stable time step
    fv_.compute_dudt(flow_, grid_, dUdt_, gas_model_, trans_prop_);

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

    // update the state
    conserved_quantities_.apply_time_derivative(dUdt_, dt_);
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
        "  step: {:>8}, t = {:.6e} ({:.1f}%), dt = {:.6e} (cfl={:.1f}), wc = {:.1f}s", step,
        t_, t_ / max_time_ * 100, dt_, dt_ / stable_dt_,  wc);
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
