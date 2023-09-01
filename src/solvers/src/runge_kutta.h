#ifndef RUNGE_KUTTA_H
#define RUNGE_KUTTA_H

#include <finite_volume/src/conserved_quantities.h>
#include <gas/src/flow_state.h>
#include "solver.h"

struct RungeKutta : public Solver {
protected:
    int take_step();
    bool print_this_step();
    bool plot_this_step();
    int plot_solution();
    int print_progress();

private:
    FlowStates<double> _flow;
    FlowStates<double> _left;
    FlowStates<double> _right;
    ConservedQuantities<double> _conserved_quantities;
    ConservedQuantities<double> _dUdt;

};

#endif
