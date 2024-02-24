#include "post.h"

#include <ibis/commands/post_commands/plot_vtk.h>
#include <ibis/config.h>
#include <spdlog/spdlog.h>

#include <Kokkos_Core.hpp>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

void setup_post_cli(CLI::App& ibis) {
    ibis.add_subcommand("post", "post-process the simulation");
}

int post(int argc, char* argv[]) {
    if (argc < 3) {
        spdlog::error("No post-processing command given");
        return 1;
    }

    std::string command = std::string(argv[2]);
    if (command == "plot_vtk") {
        std::vector<std::string> extra_vars{};
        for (int i = 3; i < argc; i++) {
            std::string arg = argv[i];
            if (arg.rfind("--add_", 0) == 0) {
                extra_vars.push_back(arg.substr(6));
            }
        }
        Kokkos::initialize(argc, argv);
        json directories = read_directories();
        plot_vtk<double>(directories, extra_vars);
        Kokkos::finalize();
    }
    return 0;
}
