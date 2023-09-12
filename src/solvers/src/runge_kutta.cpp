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

int RungeKutta::initialise() {return 0;}
int RungeKutta::finalise() {return 0;}
int RungeKutta::take_step() {return 0;}
bool RungeKutta::print_this_step() {return false;}
bool RungeKutta::plot_this_step() {return false;}
int RungeKutta::plot_solution() {return 0;}
int RungeKutta::print_progress() {return 0;}
void RungeKutta::print_stop_reason() {}
bool RungeKutta::stop_now() {return true;}
