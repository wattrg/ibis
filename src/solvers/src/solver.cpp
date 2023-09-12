#include <iostream>
#include <nlohmann/json.hpp>
#include "solver.h"

using json = nlohmann::json;

int Solver::solve() {
    initialise();
    
    for (unsigned int step = 0; step < max_step_; step++) {
        take_step();

        if (print_this_step()) {
            print_progress();
        }

        if (plot_this_step()) {
            plot_solution();
        }

        if (stop()) {
            print_stop_reason();
            break;
        }
    }

    finalise();

    return 0;
}

Solver * make_solver(json solver_config) {
    (void) solver_config;
    return NULL;
}
