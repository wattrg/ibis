#ifndef VTK_FLOW_FORMAT_H
#define VTK_FLOW_FORMAT_H

#include <io/accessor.h>
#include <io/io.h>

#include "finite_volume/finite_volume.h"
#include "gas/transport_properties.h"

template <typename T>
class VtkTextOutput : public FVOutput<T> {
public:
    VtkTextOutput();

    int write(const typename FlowStates<T>::mirror_type& fs, FiniteVolume<T>& fv,
              const GridBlock<T>& grid, const IdealGas<T>& gas_model,
              const TransportProperties<T>& trans_prop, std::string plot_dir,
              std::string time_dir, double time);

    void write_coordinating_file(std::string plot_dir);

private:
    std::vector<double> times_;
    std::vector<std::string> dirs_;
};

template <typename T>
class VtkBinaryOutput : public FVOutput<T> {
public:
    VtkBinaryOutput();

    int write(const typename FlowStates<T>::mirror_type& fs, FiniteVolume<T>& fv,
              const GridBlock<T>& grid, const IdealGas<T>& gas_model,
              const TransportProperties<T>& trans_prop, std::string plot_dir,
              std::string time_dir, double time);

    void write_coordinating_file(std::string plot_dir);

private:
    std::vector<double> times_;
    std::vector<std::string> dirs_;
};

#endif
