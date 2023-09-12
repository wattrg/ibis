#include <iostream>
#include <nlohmann/json.hpp>
#include "solver.h"

#include "runge_kutta.h"

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

        if (stop_now()) {
            print_stop_reason();
            break;
        }
    }

    finalise();

    return 0;
}

Solver * make_solver(json solver_config, GridBlock<double> grid) {
    std::string solver_name = solver_config.at("name");
    if (solver_name == "runge_kutta") {
        return new RungeKutta(solver_config, grid);
    }
    return NULL;
}
