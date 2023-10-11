#include <fstream>
#include <algorithm>
#include <spdlog/spdlog.h>
#include "native.h"

template <typename T>
int NativeOutput<T>::write(const FlowStates<T>& fs, const GridBlock<T>& grid, std::string dir, double time){
    std::ofstream meta(dir + "/meta_data");
    if (!meta) {
        spdlog::error("failed to open {}", dir+"/time");
    }
    meta << std::fixed << std::setprecision(16);
    meta << "time: " << time;
    meta.close();

    std::ofstream temp(dir + "/T");
    if (!temp) {
        spdlog::error("failed to open {}", dir+"/T");
        return 1;
    }
    temp << std::fixed << std::setprecision(16);
    for (int cell_i = 0; cell_i < grid.num_cells(); cell_i++) {
        temp << fs.gas.temp(cell_i) << std::endl;;
    }
    temp.close();

    std::ofstream pressure(dir + "/p");
    if (!pressure) {
        spdlog::error("failed to open {}", dir+"/p");
        return 1;
    }
    pressure << std::fixed << std::setprecision(16);
    for (int cell_i = 0; cell_i < grid.num_cells(); cell_i++) {
        pressure << fs.gas.pressure(cell_i) << std::endl;;
    }
    pressure.close();

    std::ofstream vx(dir + "/vx");
    if (!vx) {
        spdlog::error("failed to open {}", dir+"/vx");
        return 1;
    }
    vx << std::fixed << std::setprecision(16);
    for (int cell_i = 0; cell_i < grid.num_cells(); cell_i++) {
        vx << fs.vel.x(cell_i) << std::endl;;
    }
    vx.close();

    std::ofstream vy(dir + "/vy");
    if (!vy) {
        spdlog::error("failed to open {}", dir+"/vy");
        return 1;
    }
    vy << std::fixed << std::setprecision(16);
    for (int cell_i = 0; cell_i < grid.num_cells(); cell_i++) {
        vy << fs.vel.y(cell_i) << std::endl;;
    }
    vy.close();

    std::ofstream vz(dir + "/vz");
    if (!vz) {
        spdlog::error("failed to open {}", dir+"/vz");
        return 1;
    }
    vz << std::fixed << std::setprecision(16);
    for (int cell_i = 0; cell_i < grid.num_cells(); cell_i++) {
        vz << fs.vel.z(cell_i) << std::endl;;
    }
    vz.close();

    return 0;
}
template class NativeOutput<double>;

template <typename T>
int NativeInput<T>::read(FlowStates<T>& fs, const GridBlock<T>& grid, std::string dir){
    int num_cells = grid.num_cells();
    std::string line;
    std::ifstream temp(dir + "/T");
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

    std::ifstream pressure(dir + "/p");
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

    std::ifstream vx(dir + "/vx");
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

    std::ifstream vy(dir + "/vy");
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
template class NativeInput<double>;
