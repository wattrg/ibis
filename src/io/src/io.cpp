#include <fstream>
#include <stdexcept>
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <spdlog/spdlog.h>
#include "io.h"

template <typename T>
int write_native(const FlowStates<T>& fs, const GridBlock<T>& grid, std::string& flow_dir, int flow_i) {
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
    temp << std::fixed << std::setprecision(16);
    for (int cell_i = 0; cell_i < grid.num_cells(); cell_i++) {
        temp << fs.gas.temp(cell_i) << std::endl;;
    }
    temp.close();

    std::ofstream pressure(folder_name + "/p");
    if (!pressure) {
        spdlog::error("failed to open new pressure directory");
        return 1;
    }
    pressure << std::fixed << std::setprecision(16);
    for (int cell_i = 0; cell_i < grid.num_cells(); cell_i++) {
        pressure << fs.gas.pressure(cell_i) << std::endl;;
    }
    pressure.close();

    std::ofstream vx(folder_name + "/vx");
    if (!vx) {
        spdlog::error("failed to open new vx directory");
        return 1;
    }
    vx << std::fixed << std::setprecision(16);
    for (int cell_i = 0; cell_i < grid.num_cells(); cell_i++) {
        vx << fs.vel.x(cell_i) << std::endl;;
    }
    vx.close();

    std::ofstream vy(folder_name + "/vy");
    if (!vy) {
        spdlog::error("failed to open new vy directory");
        return 1;
    }
    vy << std::fixed << std::setprecision(16);
    for (int cell_i = 0; cell_i < grid.num_cells(); cell_i++) {
        vy << fs.vel.y(cell_i) << std::endl;;
    }
    vy.close();

    std::ofstream vz(folder_name + "/vz");
    if (!vz) {
        spdlog::error("failed to open new vz directory");
        return 1;
    }
    vz << std::fixed << std::setprecision(16);
    for (int cell_i = 0; cell_i < grid.num_cells(); cell_i++) {
        vz << fs.vel.z(cell_i) << std::endl;;
    }
    vz.close();

    return 0;
}

template <typename T>
int read_native(const FlowStates<T>& fs, const GridBlock<T>& grid, std::string flow_dir){
    int num_cells = grid.num_cells();
    std::string line;
    std::ifstream temp(flow_dir + "/T");
    if (!temp) {
        spdlog::error("Unable to load initial temperature");
        return 1;
    }
    int cell_i = 0;
    while (getline(temp, line)) {
        fs.gas.temp(cell_i) = stod(line);
        cell_i++; 
    }
    if (cell_i != num_cells){
        spdlog::error("Incorrect number of values in initial T. {} cells, but {} temperature values", num_cells, cell_i);
    }

    std::ifstream pressure(flow_dir + "/p");
    if (!pressure) {
        spdlog::error("Unable to load initial pressure");
        return 1;
    }
    cell_i = 0;
    while (getline(pressure, line)){
        fs.gas.pressure(cell_i) = stod(line);
        cell_i++;
    }
    pressure.close();

    std::ifstream vx(flow_dir + "/vx");
    if (!vx) {
        spdlog::error("Unable to load initial vx");
        return 1;
    }
    cell_i = 0;
    while (getline(vx, line)){
        fs.vel.x(cell_i) = stod(line);
        cell_i++;
    }
    vx.close();

    std::ifstream vy(flow_dir + "/vy");
    if (!vy) {
        spdlog::error("Unable to load initial vy");
        return 1;
    }
    cell_i = 0;
    while (getline(vy, line)){
        fs.vel.y(cell_i) = stod(line);
        cell_i++;
    }
    vy.close();

    for (int cell_i = 0; cell_i < num_cells; cell_i++){
        fs.gas.rho(cell_i) = fs.gas.pressure(cell_i) / (287.0 * fs.gas.temp(cell_i));
        fs.gas.energy(cell_i) = 717.5 * fs.gas.temp(cell_i);
    }

    return 0;
}

std::string pad_time_index(int time_idx, unsigned long len) {
    std::string time_index = std::to_string(time_idx);
    return std::string(len - std::min(len, time_index.length()), '0') + time_index;
}

template <typename T>
int FVIO<T>::write(const FlowStates<T>& fs, const GridBlock<T>& grid, double time) {
    (void) time;
    std::string time_index = pad_time_index(time_index_, 4);
    std::string directory_name = directory_ + "/" + time_index;
    std::filesystem::create_directory(directory_name);

    int result = 0;
    switch (format_) {
        case FlowFormat::Native:
            result = write_native(fs, grid, directory_name, time_index_);
            break;
        case FlowFormat::Vtk:
            throw std::runtime_error("Not implemented yet");
            break;
    } 
    time_index_ ++;
    return result;
}

template<typename T>
int FVIO<T>::read(FlowStates<T>& fs, const GridBlock<T>& grid, int time_idx) {
    std::string time_index = pad_time_index(time_idx, 4);
    std::string directory_name = directory_ + "/" + time_index;
    
    int result = 0;
    switch (format_) {
        case FlowFormat::Native:
            result = read_native(fs, grid, directory_name);
            break;
        case FlowFormat::Vtk:
            throw std::runtime_error("Not implemented");
            break;
    }
    return result;
}

template class FVIO<double>;

