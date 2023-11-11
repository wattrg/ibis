#include <cstdlib>
#include <fstream>
#include <Kokkos_Core.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "run.h"
#include "../../../grid/src/grid.h"
#include "../../../solvers/src/solver.h"
#include "../config.h"

using json = nlohmann::json;



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
        // we need to make the solver (and thus allocate all the kokkos memory)
        // inside a block, so that the solver (and thus all kokkos managed memory)
        // is removed before Kokkos::finalise is called
        std::unique_ptr<Solver> solver = make_solver(config, grid_dir, flow_dir);
        result = solver->solve();
    }

    Kokkos::finalize();
    
    if (result != 0) {
        spdlog::error("run failed");
    }
    else { 
        spdlog::info("run complete");
    }
    return result;
}
