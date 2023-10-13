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
std::unique_ptr<FVInput<T>> make_fv_input(FlowFormat format) {
    switch (format) {
        case FlowFormat::Native:
            return std::unique_ptr<FVInput<T>>(new NativeInput<T>());
        case FlowFormat::Vtk:
            spdlog::error("Reading VTK files not supported");
            throw std::runtime_error("Reading VTK files not supported");
        default:
            throw std::runtime_error("Unreachable");
    }
}

template <typename T>
std::unique_ptr<FVOutput<T>> make_fv_output(FlowFormat format) {
    switch (format) {
        case FlowFormat::Native:
            return std::unique_ptr<FVOutput<T>>(new NativeOutput<T>());
        case FlowFormat::Vtk:
            return std::unique_ptr<FVOutput<T>>(new VtkOutput<T>());
        default:
            throw std::runtime_error("Unreachable");
    }
}

template <typename T>
FVIO<T>::FVIO(FlowFormat input_format, FlowFormat output_format, std::string input_dir, std::string output_dir, int time_index)
    : input_(make_fv_input<T>(input_format)),
      output_(make_fv_output<T>(output_format)),
      time_index_(time_index),
      input_dir_(input_dir),
      output_dir_(output_dir)
{}

template <typename T>
FVIO<T>::FVIO(FlowFormat input, FlowFormat output) 
    : FVIO(input, output, "flow", "flow", 0) {}

template <typename T>
FVIO<T>::FVIO(FlowFormat input, FlowFormat output, std::string input_dir, std::string output_dir) 
    : FVIO(input, output, input_dir, output_dir, 0) {}


template <typename T>
FVIO<T>::FVIO() : FVIO(FlowFormat::Native, FlowFormat::Native, "flow", "flow", 0) {}

template <typename T>
int FVIO<T>::write(const FlowStates<T>& fs, const GridBlock<T>& grid, double time) {
    std::string time_index = pad_time_index(time_index_, 4);
    std::string directory_name = output_dir_ + "/" + time_index;
    std::filesystem::create_directory(output_dir_);
    std::filesystem::create_directory(directory_name);
    int result = output_->write(fs, grid, output_dir_, time_index, time);
    std::ofstream flows("config/flows", std::ios_base::app);
    flows << time << std::endl;
    time_index_ ++;
    return result;
}

template<typename T>
int FVIO<T>::read(FlowStates<T>& fs, const GridBlock<T>& grid, int time_idx) {
    std::string time_index = pad_time_index(time_idx, 4);
    std::string directory_name = input_dir_ + "/" + time_index;
    int result = input_->read(fs, grid, directory_name);
    return result;
}

template <typename T>
void FVIO<T>::write_coordinating_file() {
    output_->write_coordinating_file(output_dir_);
}

template class FVIO<double>;

