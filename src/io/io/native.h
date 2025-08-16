#ifndef NATIVE_FLOW_FORMAT_H
#define NATIVE_FLOW_FORMAT_H

#include <io/io.h>

#include "gas/transport_properties.h"

template <typename T, class MemModel>
class NativeTextInput : public FVInput<T, MemModel> {
public:
    NativeTextInput() {}

    int read(typename FlowStates<T>::mirror_type& fs, GridBlock<MemModel, T>& grid,
             const IdealGas<T>& gas_model, const TransportProperties<T>& trans_prop,
             std::string dir, json& meta_data);

    bool combined_grid_and_flow() const { return false; }
};

template <typename T, class MemModel>
class NativeTextOutput : public FVOutput<T, MemModel> {
public:
    NativeTextOutput() {}

    int write(const typename FlowStates<T>::mirror_type& fs,
              FiniteVolume<T, MemModel>& fv, const GridBlock<MemModel, T>& grid,
              const IdealGas<T>& gas_model, const TransportProperties<T>& trans_prop,
              std::string plot_dir, std::string time_dir, Ibis::real time);

    void write_coordinating_file(std::string plot_dir) { (void)plot_dir; }

    bool combined_grid_and_flow() const { return false; }
};

template <typename T, class MemModel>
class NativeBinaryInput : public FVInput<T, MemModel> {
public:
    NativeBinaryInput() {}

    int read(typename FlowStates<T>::mirror_type& fs, GridBlock<MemModel, T>& grid,
             const IdealGas<T>& gas_model, const TransportProperties<T>& trans_prop,
             std::string dir, json& meta_data);

    bool combined_grid_and_flow() const { return false; }
};

template <typename T, class MemModel>
class NativeBinaryOutput : public FVOutput<T, MemModel> {
public:
    NativeBinaryOutput() {}

    int write(const typename FlowStates<T>::mirror_type& fs,
              FiniteVolume<T, MemModel>& fv, const GridBlock<MemModel, T>& grid,
              const IdealGas<T>& gas_model, const TransportProperties<T>& trans_prop,
              std::string plot_dir, std::string time_dir, Ibis::real time);

    void write_coordinating_file(std::string plot_dir) { (void)plot_dir; }

    bool combined_grid_and_flow() const { return false; }
};

#endif
