#include "runge_kutta.h"
#include "solver.h"

RungeKutta::~RungeKutta() {}

RungeKutta::RungeKutta(json config, GridBlock<double> grid, std::string grid_dir, std::string flow_dir) 
    : Solver(grid_dir, flow_dir)
{
    // configuration
    max_time_ = config.at("max_time");
    max_step_ = config.at("max_step");
    print_frequency_ = config.at("print_frequency");
    plot_frequency_ = config.at("plot_frequency");
    plot_every_n_steps_ = config.at("plot_every_n_steps");
    cfl_ = config.at("cfl"); 

    // memory
    grid_ = grid;
    flow_ = FlowStates<double>(grid_.num_cells());
    conserved_quantities_ = ConservedQuantities<double>(grid_.num_cells(), grid_.dim());
    dUdt_ = ConservedQuantities<double>(grid_.num_cells(), grid_.dim());
    fv_ = FiniteVolume<double>(grid_);

    // progress
    time_since_last_plot_ = 0.0;
    n_solutions_ = 0;
    t_ = 0.0;
}

int RungeKutta::initialise() {
    read_initial_condition(flow_, flow_dir_);
    return 0;
}

int RungeKutta::finalise() {
    return 0;
}

int RungeKutta::take_step() {
    fv_.compute_dudt(flow_, grid_, dUdt_);
    double dt = cfl_ / fv_.estimate_signal_frequency(flow_, grid_);
    conserved_quantities_.apply_time_derivative(dUdt_, dt);
    t_ += dt;
    time_since_last_plot_ += dt;
    return 0;
}
bool RungeKutta::print_this_step(unsigned int step) {
    if (step % print_frequency_ == 0) return true;
    return false;
}

bool RungeKutta::plot_this_step(unsigned int step) {
    if (plot_every_n_steps_ > 0 && step % plot_every_n_steps_ == 0) return true;
    if (time_since_last_plot_ > plot_frequency_) return true;
    return false;
}

int RungeKutta::plot_solution(unsigned int step) {
    (void) step;
    int result =  write_flow_solution<double>(flow_, flow_dir_, n_solutions_);
    n_solutions_ ++;
    time_since_last_plot_ = 0.0;
    return result;
}

std::string RungeKutta::progress_string(unsigned int step) {
    return "  step: " + std::to_string(step) + ", t = " + std::to_string(t_);
}
std::string RungeKutta::stop_reason(unsigned int step) {
    if (t_ > max_time_) return "reached max time"; 
    if (step > max_step_) return "reached max step";
    return "Shouldn't reach here";
}
bool RungeKutta::stop_now(unsigned int step) {
    if (step >= max_step_) return true;
    if (t_ >= max_time_) return true;
    return false;
}
