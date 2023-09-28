#include <iostream>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "solver.h"
#include "runge_kutta.h"

using json = nlohmann::json;

Solver::Solver(std::string grid_dir, std::string flow_dir) 
    : grid_dir_(grid_dir), flow_dir_(flow_dir) {}

int Solver::solve() {
    initialise();
    for (int step = 0; step < max_step(); step++) {
        int result = take_step();

        if (result != 0) {
            spdlog::error("step {} failed", step);
            plot_solution(step);
            return 1;
        }

        if (print_this_step(step)) {
            std::string progress = progress_string(step);
            spdlog::info(progress);
        }

        if (plot_this_step(step)) {
            plot_solution(step);
        }

        if (stop_now(step)) {
            std::string reason = stop_reason(step);
            spdlog::info("STOPPING: {}", reason);
            plot_solution(step);
            break;
        }
    }
    finalise();

    return 0;
}

Solver * make_solver(json config, std::string grid_dir, std::string flow_dir) {
    std::string grid_file = grid_dir + "/block_0000.su2";
    json solver_config = config.at("solver");
    json grid_config = config.at("grid");
    std::string solver_name = solver_config.at("name");
    if (solver_name == "runge_kutta") {
        GridBlock<double> grid = GridBlock<double>(grid_file, grid_config);
        return new RungeKutta(solver_config, grid, grid_dir, flow_dir);
    }
    return NULL;
}

template<typename T>
int read_initial_condition(FlowStates<T>& fs, std::string flow_dir) {
    std::string line;
    std::ifstream temp(flow_dir + "/0000/T");
    int cell_i = 0;
    while (getline(temp, line)) {
        fs.gas.temp(cell_i) = stoi(line);
        cell_i++; 
    }
    temp.close();

    std::ifstream pressure(flow_dir + "/0000/p");
    cell_i = 0;
    while (getline(temp, line)){
        fs.gas.pressure(cell_i) = stoi(line);
        cell_i++;
    }
    pressure.close();

    std::ifstream vx(flow_dir + "/0000/vx");
    cell_i = 0;
    while (getline(temp, line)){
        fs.vel.x(cell_i) = stoi(line);
        cell_i++;
    }
    vx.close();

    std::ifstream vy(flow_dir + "/0000/vy");
    cell_i = 0;
    while (getline(temp, line)){
        fs.vel.y(cell_i) = stoi(line);
        cell_i++;
    }
    vy.close();
    return 0;
}

template int read_initial_condition<double>(FlowStates<double>&, std::string);

template<typename T>
int write_flow_solution(FlowStates<T>& fs, std::string flow_dir, int flow_i) {
    std::ofstream temp(flow_dir + "/" + std::to_string(flow_i) + "/T");
    for (unsigned int cell_i = 0; cell_i < fs.number_flow_states(); cell_i++) {
        temp << fs.gas.temp(cell_i);
    }
    return 0;
}

template int write_flow_solution<double>(FlowStates<double>&, std::string, int);
