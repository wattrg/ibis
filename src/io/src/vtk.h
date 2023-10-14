#ifndef VTK_FLOW_FORMAT_H
#define VTK_FLOW_FORMAT_H

#include "accessor.h"
#include "io.h"

template <typename T>
class VtkOutput : public FVOutput<T> {
public:
    VtkOutput();

    int write(const FlowStates<T>& fs, 
              const GridBlock<T>& grid, 
              std::string plot_dir, 
              std::string time_dir, 
              double time);

    void write_coordinating_file(std::string plot_dir);

private:
    std::vector<double> times_;
    std::vector<std::string> dirs_;
    std::map<std::string, std::shared_ptr<ScalarAccessor<T>>> m_scalar_accessors;
};

#endif 
