#include <iostream>
#include <filesystem>
#include <algorithm>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "solver.h"
#include "runge_kutta.h"

using json = nlohmann::json;

Solver::Solver(std::string grid_dir, std::string flow_dir) 
    : grid_dir_(grid_dir), flow_dir_(flow_dir) {}

int Solver::solve() {
    int success = initialise();
    if (success != 0) {
        spdlog::error("Failed to initialise runge kutta solver");
        return success;
    }
    for (int step = 0; step < max_step(); step++) {
        int result = take_step();
        int bad_cells = count_bad_cells();
        if (bad_cells > 0) {
            spdlog::error("Encountered {} bad cells", bad_cells);
            plot_solution(step);
            return 1;
        }

        if (result != 0) {
            spdlog::error("step {} failed", step);
            plot_solution(step);
            return 1;
        }

        if (print_this_step(step)) {
            print_progress(step);
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
        return new RungeKutta(config, grid, grid_dir, flow_dir);
    }
    return NULL;
}

template<typename T>
int read_initial_condition(FlowStates<T>& fs, std::string flow_dir) {
    std::string line;
    std::ifstream temp(flow_dir + "/0000/T");
    if (!temp) {
        spdlog::error("Unable to load initial temperature");
        return 1;
    }
    int cell_i = 0;
    while (getline(temp, line)) {
        fs.gas.temp(cell_i) = stoi(line);
        cell_i++; 
    }

    std::ifstream pressure(flow_dir + "/0000/p");
    if (!pressure) {
        spdlog::error("Unable to load initial pressure");
        return 1;
    }
    cell_i = 0;
    while (getline(pressure, line)){
        fs.gas.pressure(cell_i) = stoi(line);
        cell_i++;
    }
    pressure.close();

    std::ifstream vx(flow_dir + "/0000/vx");
    if (!vx) {
        spdlog::error("Unable to load initial vx");
        return 1;
    }
    cell_i = 0;
    while (getline(vx, line)){
        fs.vel.x(cell_i) = stoi(line);
        cell_i++;
    }
    vx.close();

    std::ifstream vy(flow_dir + "/0000/vy");
    if (!vy) {
        spdlog::error("Unable to load initial vy");
        return 1;
    }
    cell_i = 0;
    while (getline(vy, line)){
        fs.vel.y(cell_i) = stoi(line);
        cell_i++;
    }
    vy.close();

    int n_cells = cell_i;
    for (int cell_i = 0; cell_i < n_cells; cell_i++){
        fs.gas.rho(cell_i) = fs.gas.pressure(cell_i) / (287.0 * fs.gas.temp(cell_i));
        fs.gas.energy(cell_i) = 0.7171 * fs.gas.temp(cell_i);
    }

    return 0;
}

template int read_initial_condition<double>(FlowStates<double>&, std::string);

template<typename T>
int write_flow_solution(const FlowStates<T>& fs, const GridBlock<T>& grid, const std::string flow_dir, int flow_i) {
    std::string flow_index = std::to_string(flow_i);
    unsigned long len = 4;
    flow_index = std::string(len - std::min(len, flow_index.length()), '0') + flow_index;
    std::string folder_name = flow_dir + "/" + flow_index;
    std::filesystem::create_directory(folder_name);

    std::ofstream temp(folder_name + "/T");
    if (!temp) {
        spdlog::error("failed to open new temperature directory");
        return 1;
    }
    for (int cell_i = 0; cell_i < grid.num_cells(); cell_i++) {
        temp << fs.gas.temp(cell_i) << std::endl;;
    }
    temp.close();

    std::ofstream pressure(folder_name + "/p");
    if (!pressure) {
        spdlog::error("failed to open new pressure directory");
        return 1;
    }
    for (int cell_i = 0; cell_i < grid.num_cells(); cell_i++) {
        pressure << fs.gas.pressure(cell_i) << std::endl;;
    }
    pressure.close();

    std::ofstream vx(folder_name + "/vx");
    if (!vx) {
        spdlog::error("failed to open new vx directory");
        return 1;
    }
    for (int cell_i = 0; cell_i < grid.num_cells(); cell_i++) {
        vx << fs.vel.x(cell_i) << std::endl;;
    }
    vx.close();

    std::ofstream vy(folder_name + "/vy");
    if (!vy) {
        spdlog::error("failed to open new vy directory");
        return 1;
    }
    for (int cell_i = 0; cell_i < grid.num_cells(); cell_i++) {
        vy << fs.vel.y(cell_i) << std::endl;;
    }
    vy.close();

    std::ofstream vz(folder_name + "/vz");
    if (!vz) {
        spdlog::error("failed to open new vz directory");
        return 1;
    }
    for (int cell_i = 0; cell_i < grid.num_cells(); cell_i++) {
        vz << fs.vel.z(cell_i) << std::endl;;
    }
    vz.close();

    return 0;
}

template int write_flow_solution<double>(const FlowStates<double>&, const GridBlock<double>&, std::string, int);
