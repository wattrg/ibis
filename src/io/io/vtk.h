#ifndef VTK_FLOW_FORMAT_H
#define VTK_FLOW_FORMAT_H

#include <io/accessor.h>
#include <io/io.h>

#include "finite_volume/finite_volume.h"
#include "gas/transport_properties.h"

template <typename T>
class VtkOutput : public FVOutput<T> {
public:
    VtkOutput();

    int write(const typename FlowStates<T>::mirror_type& fs, FiniteVolume<T>& fv,
              const GridBlock<T>& grid, const IdealGas<T>& gas_model,
              const TransportProperties<T>& trans_prop,
              std::string plot_dir, std::string time_dir, double time);

    void write_coordinating_file(std::string plot_dir);

private:
    std::vector<double> times_;
    std::vector<std::string> dirs_;
    // std::map<std::string, std::shared_ptr<ScalarAccessor<T>>>
    //     m_scalar_accessors;
    // std::map<std::string, std::shared_ptr<VectorAccessor<T>>>
    //     m_vector_accessors;
};

#endif
