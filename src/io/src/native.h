#ifndef NATIVE_FLOW_FORMAT_H
#define NATIVE_FLOW_FORMAT_H

#include "io.h"

template <typename T>
int write_native(const FlowStates<T>& fs, const GridBlock<T>& grid, std::string dir);

template <typename T>
int read_native(FlowStates<T>& fs, const GridBlock<T>& grid, std::string dir);

#endif
