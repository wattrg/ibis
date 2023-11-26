#include <iostream>
#include <filesystem>
#include <algorithm>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "solver.h"
#include "runge_kutta/runge_kutta.h"

using json = nlohmann::json;

Solver::Solver(std::string grid_dir, 
               std::string flow_dir, 
               Units units) 
    : grid_dir_(grid_dir), flow_dir_(flow_dir), units_(units) {}

int Solver::solve() {
    int success = initialise();
    if (success != 0) {
        spdlog::error("Failed to initialise solver");
        return success;
    }

    for (int step = 0; step < max_step(); step++) {
        int result = take_step();
        if (result != 0) {
            spdlog::error("step {} failed", step);
            plot_solution(step);
            return 1;
        }

        int bad_cells = count_bad_cells();
        if (bad_cells > 0) {
            spdlog::error("Encountered {} bad cells on step {}", 
                          bad_cells, step);
            plot_solution(step);
            return 1;
        }

        if (stop_now(step)) {
            std::string reason = stop_reason(step);
            spdlog::info("STOPPING: {}", reason);
            plot_solution(step);
            break;
        }

        if (print_this_step(step)) {
            print_progress(step);
        }

        if (plot_this_step(step)) {
            plot_solution(step);
        }
    }
    finalise();

    return 0;
}

std::unique_ptr<Solver> make_solver(json config, 
                                    std::string grid_dir, 
                                    std::string flow_dir) 
{
    std::string grid_file = grid_dir + "/block_0000.su2";
    json solver_config = config.at("solver");
    json grid_config = config.at("grid");
    std::string solver_name = solver_config.at("name");

    if (solver_name == "runge_kutta") {
        GridBlock<double> grid(grid_file, grid_config);
        return std::unique_ptr<Solver>(
            new RungeKutta(config, grid, grid_dir, flow_dir)
        );
    }

    return NULL;
}
