#include <fstream>
#include <stdexcept>
#include <filesystem>
#include "io.h"

int write_native(std::string& directory_name, FlowStates<double>& flow_state) {
    int num_cells = flow_state.number_flow_states();

    // temperature
    std::ofstream temp (directory_name + "temp");
    for (int i = 0; i < num_cells; i++) {
        temp << flow_state.gas.temp(i) << "\n";
    }
    temp.close();

    // density
    std::ofstream density(directory_name + "density");
    for (int i = 0; i < num_cells; i++) {
        density << flow_state.gas.rho(i) << "\n";
    }
    density.close();
    return 0; 
}

template <>
int FVIO<double>::write(FlowStates<double>& flow_state, double time) {
    (void) time;
    std::string directory_name = directory_ +
                                "/" + 
                                std::to_string(time_index_) +
                                "block_" + 
                                std::to_string(block_index_);

    std::filesystem::create_directory(directory_name);

    int result;
    switch (format_) {
        case FlowFormat::Native:
            result = write_native(directory_name, flow_state);
            break;
        case FlowFormat::Vtk:
            throw std::runtime_error("Not implemented yet");
            break;
    } 
    return result;
}

