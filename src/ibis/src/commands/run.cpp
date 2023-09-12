#include <cstdlib>
#include <fstream>
#include <Kokkos_Core.hpp>
#include <nlohmann/json.hpp>

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
        std::cerr << "Unable to open config file. Make sure simulation is prepped\n";
        throw std::runtime_error("Unable to open config file");
    }
    json config = json::parse(f);
    f.close();
    return config;
}

GridBlock<double> read_grid(json& directories) {
    std::string grid_dir = directories.at("grid_dir");
    std::string grid_file = grid_dir + "/" + "block_0000.su2";
    return GridBlock<double> (grid_file);
}

int run(int argc, char* argv[]) {
    json directories = read_directories();
    json config = read_config(directories);
    Kokkos::initialize(argc, argv);
    
    {
        GridBlock<double> grid = read_grid(directories);
        Solver * solver = make_solver(config.at("solver"), grid);
        delete solver;
    }
    
    Kokkos::finalize();
    return 0;
}
