
#include <io/io.h>
#include <io/native.h>
#include <io/vtk.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

std::string pad_time_index(int time_idx, unsigned long len) {
    std::string time_index = std::to_string(time_idx);
    unsigned long extra_chars =
        len - std::min<unsigned long>(len, time_index.length());
    std::string padded_str = std::string(extra_chars, '0');
    return padded_str + time_index;
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
FVIO<T>::FVIO(FlowFormat input_format, FlowFormat output_format,
              std::string input_dir, std::string output_dir, int time_index)
    : input_(make_fv_input<T>(input_format)),
      output_(make_fv_output<T>(output_format)),
      time_index_(time_index),
      input_dir_(input_dir),
      output_dir_(output_dir) {}

template <typename T>
FVIO<T>::FVIO(FlowFormat input, FlowFormat output)
    : FVIO(input, output, "flow", "flow", 0) {}

template <typename T>
FVIO<T>::FVIO(FlowFormat input, FlowFormat output, std::string input_dir,
              std::string output_dir)
    : FVIO(input, output, input_dir, output_dir, 0) {}

template <typename T>
FVIO<T>::FVIO()
    : FVIO(FlowFormat::Native, FlowFormat::Native, "flow", "flow", 0) {}

template <typename T>
FVIO<T>::FVIO(int time_index)
    : FVIO(FlowFormat::Native, FlowFormat::Native, "flow", "flow", time_index) {
}

template <typename T>
int FVIO<T>::write(const FlowStates<T>& fs, const GridBlock<T>& grid,
                   const IdealGas<T>& gas_model, double time) {
    // get a copy of the flow states and grid on the CPU
    auto grid_host = grid.host_mirror();
    auto fs_host = fs.host_mirror();
    grid_host.deep_copy(grid);
    fs_host.deep_copy(fs);

    std::string time_index = pad_time_index(time_index_, 4);
    std::string directory_name = output_dir_ + "/" + time_index;
    std::filesystem::create_directory(output_dir_);
    std::filesystem::create_directory(directory_name);
    int result = output_->write(fs_host, grid_host, gas_model, output_dir_,
                                time_index, time);
    time_index_++;
    return result;
}

template <typename T>
int FVIO<T>::read(FlowStates<T>& fs, const GridBlock<T>& grid,
                  const IdealGas<T>& gas_model, json& meta_data, int time_idx) {
    auto grid_host = grid.host_mirror();
    auto fs_host = fs.host_mirror();
    std::string time_index = pad_time_index(time_idx, 4);
    std::string directory_name = input_dir_ + "/" + time_index;
    int result =
        input_->read(fs_host, grid_host, gas_model, directory_name, meta_data);
    fs.deep_copy(fs_host);
    return result;
}

template <typename T>
void FVIO<T>::write_coordinating_file() {
    output_->write_coordinating_file(output_dir_);
}

template class FVIO<double>;
