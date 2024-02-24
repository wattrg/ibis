#ifndef NATIVE_FLOW_FORMAT_H
#define NATIVE_FLOW_FORMAT_H

#include <io/io.h>

template <typename T>
class NativeInput : public FVInput<T> {
public:
    NativeInput() {}

    int read(typename FlowStates<T>::mirror_type& fs,
             const GridBlock<T>& grid,
             const IdealGas<T>& gas_model, std::string dir, json& meta_data);
};

template <typename T>
class NativeOutput : public FVOutput<T> {
public:
    NativeOutput() {}

    int write(const typename FlowStates<T>::mirror_type& fs,
              FiniteVolume<T>& fv,
              const GridBlock<T>& grid,
              const IdealGas<T>& gas_model, std::string plot_dir,
              std::string time_dir, double time);

    void write_coordinating_file(std::string plot_dir) { (void)plot_dir; }
};

#endif
