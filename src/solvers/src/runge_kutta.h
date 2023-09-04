#ifndef RUNGE_KUTTA_H
#define RUNGE_KUTTA_H

#include <finite_volume/src/conserved_quantities.h>
#include <gas/src/flow_state.h>
#include "finite_volume/src/finite_volume.h"
#include "solver.h"

class RungeKutta : public Solver {
private:
    // configuration
    double t_;
    double max_time_;
    int print_frequency_;
    double plot_energy_n_steps_;
    double plot_frequency_;
    double time_since_last_plot_;
    int n_solutions_;
    double cfl_;

private:
    // implementation
    int take_step();
    bool print_this_step();
    bool plot_this_step();
    int plot_solution();
    int print_progress();

private:
    // memory
    FlowStates<double> flow_;
    ConservedQuantities<double> conserved_quantities_;
    ConservedQuantities<double> dUdt_;

private:
    // spatial discretisation
    FiniteVolume<double> fv_;
};

#endif
