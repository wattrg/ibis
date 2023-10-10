#ifndef VTK_FLOW_FORMAT_H
#define VTK_FLOW_FORMAT_H

#include "io.h"

template <typename T>
int write_vtk(const FlowStates<T>& fs, const GridBlock<T>& grid, std::string dir);

#endif 
