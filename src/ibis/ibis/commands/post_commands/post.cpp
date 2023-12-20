#include "post.h"

#include <ibis/commands/post_commands/plot_vtk.h>
#include <ibis/config.h>
#include <spdlog/spdlog.h>

#include <Kokkos_Core.hpp>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int post(int argc, char* argv[]) {
    if (argc < 3) {
        spdlog::error("No post-processing command given");
        return 1;
    }

    std::string command = std::string(argv[2]);
    if (command == "plot_vtk") {
        Kokkos::initialize(argc, argv);
        json directories = read_directories();
        plot_vtk<double>(directories);
        Kokkos::finalize();
    }
    return 0;
}
