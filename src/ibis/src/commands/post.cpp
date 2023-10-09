#include <spdlog/spdlog.h>
#include "post.h"

int post(int argc, char* argv[]) {
    if (argc < 3) {
        spdlog::error("No post-processing command given");
    }

    std::string command = std::string(argv[2]);
    if (command == "plot_vtk"){
        
    }
    return 0;
}
