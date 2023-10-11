#ifndef VTK_FLOW_FORMAT_H
#define VTK_FLOW_FORMAT_H

#include "io.h"

template <typename T>
class VtkOutput : public FVOutput<T> {
public:
    int write(const FlowStates<T>& fs, const GridBlock<T>& grid, std::string dir, double time);

};

#endif 
