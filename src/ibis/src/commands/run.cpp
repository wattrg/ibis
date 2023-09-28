#include <cstdlib>
#include <fstream>
#include <Kokkos_Core.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "run.h"
#include "../../../grid/src/grid.h"
#include "../../../solvers/src/solver.h"

using json = nlohmann::json;

json read_directories() {
    std::string ibis = std::getenv("IBIS");
    std::ifstream f(ibis + "/resources/defaults/directories.json");
    json directories = json::parse(f);
    f.close();
    return directories;
}

json read_config(json& directories) {
    // read the config file
    std::string config_dir = directories.at("config_dir");
    std::string config_file = directories.at("config_file");
    std::ifstream f(config_dir + "/" + config_file);
    if (!f) {
        spdlog::error("Unable to open config file. Make sure simulation is prepped");
        throw std::runtime_error("Unable to open config file");
    }
    json config = json::parse(f);
    f.close();
    return config;
}

// GridBlock<double> read_grid(json& directories) {
//     std::string grid_dir = directories.at("grid_dir");
//     std::string grid_file = grid_dir + "/" + "block_0000.su2";
//     return GridBlock<double> (grid_file);
// }

void print_config_info(json config) {
    spdlog::info("solver: {}", std::string(config.at("solver").at("name")));
    spdlog::info("flux calculator: {}", std::string(config.at("convective_flux").at("flux_calculator")));
}

int run(int argc, char* argv[]) {
    json directories = read_directories();
    json config = read_config(directories);

    print_config_info(config);

    std::string grid_dir = directories.at("grid_dir");
    std::string flow_dir = directories.at("flow_dir");
    Kokkos::initialize(argc, argv);
    int result; 
    {
        // GridBlock<double> grid = read_grid(directories);
        Solver * solver = make_solver(config, grid_dir, flow_dir);
        result = solver->solve();
        delete solver;
    }

    Kokkos::finalize();
    
    if (result != 0) {
        spdlog::error("run failed");
    }
    
    spdlog::info("run complete");
    return 0;
}