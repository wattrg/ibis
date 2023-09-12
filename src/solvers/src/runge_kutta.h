#ifndef RUNGE_KUTTA_H
#define RUNGE_KUTTA_H

// #include <finite_volume/src/conserved_quantities.h>
// #include <gas/src/flow_state.h>
#include "../../finite_volume/src/conserved_quantities.h"
#include "../../gas/src/flow_state.h"
// #include "finite_volume/src/finite_volume.h"
#include "../../grid/src/grid.h"
#include "solver.h"


class RungeKutta : public Solver {
public:
    RungeKutta(json config, GridBlock<double> grid);
    virtual ~RungeKutta();

private:
    // configuration
    double max_time_;
    int max_step_;
    int print_frequency_;
    double plot_frequency_;
    double plot_every_n_steps_;
    double cfl_;

private:
    // progress
    double time_since_last_plot_;
    int n_solutions_;
    double t_;

private:
    // implementation
    int initialise();
    int finalise();
    int take_step();
    bool print_this_step();
    bool plot_this_step();
    int plot_solution();
    int print_progress();
    void print_stop_reason();
    bool stop_now();

private:
    // memory
    GridBlock<double> grid_;
    FlowStates<double> flow_;
    ConservedQuantities<double> conserved_quantities_;
    ConservedQuantities<double> dUdt_;

private:
    // spatial discretisation
    FiniteVolume<double> fv_;
};

#endif
