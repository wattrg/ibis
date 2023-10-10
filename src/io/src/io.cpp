#include <fstream>
#include <stdexcept>
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <spdlog/spdlog.h>
#include "io.h"
#include "native.h"
#include "vtk.h"


std::string pad_time_index(int time_idx, unsigned long len) {
    std::string time_index = std::to_string(time_idx);
    return std::string(len - std::min(len, time_index.length()), '0') + time_index;
}

template <typename T>
int FVIO<T>::write(const FlowStates<T>& fs, const GridBlock<T>& grid, double time) {
    (void) time;
    std::string time_index = pad_time_index(time_index_, 4);
    std::string directory_name = output_dir_ + "/" + time_index;
    std::filesystem::create_directory(directory_name);

    int result = 0;
    switch (output_format_) {
        case FlowFormat::Native:
            result = write_native(fs, grid, directory_name);
            break;
        case FlowFormat::Vtk:
            result = write_vtk(fs, grid, directory_name);
            break;
    } 
    time_index_ ++;
    return result;
}

template<typename T>
int FVIO<T>::read(FlowStates<T>& fs, const GridBlock<T>& grid, int time_idx) {
    std::string time_index = pad_time_index(time_idx, 4);
    std::string directory_name = input_dir_ + "/" + time_index;
    
    int result = 0;
    switch (input_format_) {
        case FlowFormat::Native:
            result = read_native(fs, grid, directory_name);
            break;
        case FlowFormat::Vtk:
            spdlog::error("Reading VTK files not supported");
            throw std::runtime_error("Not implemented");
            break;
    }
    return result;
}

template class FVIO<double>;

