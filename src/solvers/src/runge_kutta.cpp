#include <spdlog/spdlog.h>
#include "runge_kutta.h"
#include "solver.h"
#include "../../finite_volume/src/flow_state_conserved_conversion.h"


RungeKutta::RungeKutta(json config, GridBlock<double> grid, std::string grid_dir, std::string flow_dir) 
    : Solver(grid_dir, flow_dir)
{
    // configuration
    json solver_config = config.at("solver");
    max_time_ = solver_config.at("max_time");
    max_step_ = solver_config.at("max_step");
    print_frequency_ = solver_config.at("print_frequency");
    plot_frequency_ = solver_config.at("plot_frequency");
    plot_every_n_steps_ = solver_config.at("plot_every_n_steps");
    cfl_ = solver_config.at("cfl"); 

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
    n_solutions_ = 0;
    t_ = 0.0;
}

int RungeKutta::initialise() {
    return read_initial_condition(flow_, flow_dir_);
}

int RungeKutta::finalise() {
    return 0;
}

int RungeKutta::take_step() {
    flow_states_to_conserved(conserved_quantities_, flow_);
    double dt = cfl_ / fv_.estimate_signal_frequency(flow_, grid_);
    fv_.compute_dudt(flow_, grid_, dUdt_);
    conserved_quantities_.apply_time_derivative(dUdt_, dt);
    conserved_to_flow_states(conserved_quantities_, flow_);

    t_ += dt;
    time_since_last_plot_ += dt;
    return 0;
}
bool RungeKutta::print_this_step(unsigned int step) {
    if (step != 0 && step % print_frequency_ == 0) return true;
    return false;
}

bool RungeKutta::plot_this_step(unsigned int step) {
    if (plot_every_n_steps_ > 0 && step != 0 && step % plot_every_n_steps_ == 0) return true;
    if (plot_frequency_ > 0 && time_since_last_plot_ > plot_frequency_) return true;
    return false;
}

int RungeKutta::plot_solution(unsigned int step) {
    n_solutions_ ++;
    int result =  write_flow_solution<double>(flow_, grid_, flow_dir_, n_solutions_);
    time_since_last_plot_ = 0.0;
    spdlog::info("    written flow solution step {}", step);
    return result;
}

std::string RungeKutta::progress_string(unsigned int step) {
    return "  step: " + std::to_string(step) + ", t = " + std::to_string(t_);
}
std::string RungeKutta::stop_reason(unsigned int step) {
    if (t_ > max_time_) return "reached max time"; 
    if (step >= max_step_-1) return "reached max step";
    return "Shouldn't reach here";
}
bool RungeKutta::stop_now(unsigned int step) {
    if (step >= max_step_-1) return true;
    if (t_ >= max_time_) return true;
    return false;
}
