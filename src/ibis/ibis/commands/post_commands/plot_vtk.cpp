
#include <gas/flow_state.h>
#include <gas/transport_properties.h>
#include <grid/grid.h>
#include <ibis/commands/post_commands/plot_vtk.h>
#include <ibis/config.h>
#include <io/io.h>
#include <spdlog/spdlog.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "finite_volume/finite_volume.h"

template <typename T>
void plot_vtk(json directories, std::vector<std::string> extra_vars) {
    std::string flow_dir = directories.at("flow_dir");
    std::string grid_dir = directories.at("grid_dir");
    std::string plot_dir = directories.at("plot_dir");

    // read the flows file to figure out what flow files exist
    std::ifstream flows(flow_dir + "/flows");
    std::vector<std::string> dirs;
    std::string line;
    while (std::getline(flows, line)) {
        dirs.push_back(line);
    }
    json config = read_config(directories);
    IdealGas<double> gas_model{config.at("gas_model")};

    FVIO<T> io(FlowFormat::Native, FlowFormat::Vtk, flow_dir, plot_dir);
    for (auto& extra_var : extra_vars) {
        io.add_output_variable(extra_var);
    }

    GridBlock<T> grid(grid_dir + "/block_0000.su2", config.at("grid"));
    FiniteVolume<T> fv(grid, config);
    FlowStates<T> fs(grid.num_cells());
    for (unsigned int time_idx = 0; time_idx < dirs.size(); time_idx++) {
        json meta_data;
        io.read(fs, grid, gas_model, meta_data, time_idx);
        io.write(fs, fv, grid, gas_model, meta_data.at("time"));
        spdlog::info("Written VTK file at time index {}", time_idx);
    }
    io.write_coordinating_file();
}
template void plot_vtk<double>(json, std::vector<std::string>);
