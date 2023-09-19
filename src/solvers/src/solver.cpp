#include <iostream>
#include <nlohmann/json.hpp>
#include "solver.h"

#include "runge_kutta.h"

using json = nlohmann::json;

Solver::Solver(std::string grid_dir, std::string flow_dir) 
    : grid_dir_(grid_dir), flow_dir_(flow_dir) {}

int Solver::solve() {
    initialise();
    std::cout << "Beginning solve" << std::endl; 
    for (unsigned int step = 0; step < max_step_; step++) {
        take_step();

        if (print_this_step(step)) {
            print_progress(step);
        }

        if (plot_this_step(step)) {
            plot_solution(step);
        }

        if (stop_now(step)) {
            std::string reason = stop_reason(step);
            std::cout << "STOPPING: " << reason << std::endl;
            break;
        }
    }
    finalise();

    return 0;
}

Solver * make_solver(json solver_config, std::string grid_dir, std::string flow_dir) {
    std::string solver_name = solver_config.at("name");
    if (solver_name == "runge_kutta") {
        return new RungeKutta(solver_config, grid_dir, flow_dir);
    }
    return NULL;
}
