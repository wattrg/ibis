#ifndef FV_IO_H
#define FV_IO_H

#include <string>
#include "../../src/gas/src/flow_state.h"
#include "../../src/grid/src/grid.h"

enum class FlowFormat {
    Native, Vtk
};

template <typename T>
class FVIO {
public:
    FVIO(FlowFormat input_format, FlowFormat output_format, int time_index) 
        : input_format_(input_format), 
          output_format_(output_format),
          time_index_(time_index) {}

    FVIO(FlowFormat input, FlowFormat output) : FVIO(input, output, 0) {}

    FVIO(FlowFormat format) : FVIO(format, format, 0) {}

    FVIO() : FVIO(FlowFormat::Native, FlowFormat::Native, 0) {}

    int read(FlowStates<T>& fs, const GridBlock<T>& grid, int time_idx);
    int write(const FlowStates<T>& flow_state, const GridBlock<T>& grid, double time);

private:
    FlowFormat input_format_;
    FlowFormat output_format_;
    int time_index_ = 0;
    std::string directory_ = "flow";
};

#endif
