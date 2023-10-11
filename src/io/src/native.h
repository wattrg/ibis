#ifndef NATIVE_FLOW_FORMAT_H
#define NATIVE_FLOW_FORMAT_H

#include "io.h"

template <typename T>
class NativeInput : public FVInput<T> {
public:
    NativeInput() {}

    int read(FlowStates<T>& fs, const GridBlock<T>& grid, std::string dir);
};

template <typename T>
class NativeOutput : public FVOutput<T> {
public:
    NativeOutput() {}

    int write(const FlowStates<T>& fs, const GridBlock<T>& grid, std::string dir, double time);
};

#endif
