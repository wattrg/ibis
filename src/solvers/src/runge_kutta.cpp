#include "runge_kutta.h"

RungeKutta::~RungeKutta() {}

RungeKutta::RungeKutta(json config, GridBlock<double> grid) 
    : grid_(grid), 
      flow_(FlowStates<double>(grid.num_cells())),
      conserved_quantities_(ConservedQuantities<double>(grid.num_cells(), grid.dim())),
      dUdt_(ConservedQuantities<double>(grid.num_cells(), grid.dim())),
      fv_(FiniteVolume<double>(grid))
{
    t_ = 0.0;
    max_time_ = config.at("max_time");
    max_step_ = config.at("max_step");
    print_frequency_ = config.at("print_frequency");
    plot_frequency_ = config.at("plot_frequency");
    plot_every_n_steps_ = config.at("plot_every_n_steps");
    cfl_ = config.at("cfl"); 

    time_since_last_plot_ = 0.0;
    n_solutions_ = 0;
    t_ = 0.0;

}

int RungeKutta::initialise() {
    std::cout << "Initialising runge kutta solver... ";
    // read initial condition
    std::cout << "done" << std::endl;
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
    return 0;
}

int RungeKutta::print_progress(unsigned int step) {
    std::cout << "  step: " << step << ", t = " << t_ << std::endl;
    return 0;
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
