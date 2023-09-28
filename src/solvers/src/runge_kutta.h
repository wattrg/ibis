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
    RungeKutta(json config, GridBlock<double> grid, std::string grid_dir, std::string flow_dir);

    ~RungeKutta();

    int solver();


private:
    // configuration
    double max_time_;
    unsigned int max_step_;
    unsigned int print_frequency_;
    double plot_frequency_;
    int plot_every_n_steps_;
    double cfl_;
    std::string grid_dir_;
    std::string flow_dir_;

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
    bool print_this_step(unsigned int step);
    bool plot_this_step(unsigned int step);
    int plot_solution(unsigned int step);
    std::string progress_string(unsigned int step);
    std::string stop_reason(unsigned int step);
    bool stop_now(unsigned int step);
    int max_step() const {return max_step_;}

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
