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
    FVIO(FlowFormat input_format, FlowFormat output_format, std::string input_dir, std::string output_dir, int time_index) 
        : input_format_(input_format), 
          output_format_(output_format),
          time_index_(time_index),
          input_dir_(input_dir),
          output_dir_(output_dir)
    {}

    FVIO(FlowFormat input, FlowFormat output) : FVIO(input, output, "flow", "flow", 0) {}

    FVIO(FlowFormat input, FlowFormat output, std::string input_dir, std::string output_dir) 
        : FVIO(input, output, input_dir, output_dir, 0) {}

    FVIO(FlowFormat format) : FVIO(format, format, "flow", "flow", 0) {}

    FVIO() : FVIO(FlowFormat::Native, FlowFormat::Native, "flow", "flow", 0) {}

    int read(FlowStates<T>& fs, const GridBlock<T>& grid, int time_idx);

    int write(const FlowStates<T>& flow_state, const GridBlock<T>& grid, double time);

private:
    FlowFormat input_format_;
    FlowFormat output_format_;
    int time_index_;
    std::string input_dir_;
    std::string output_dir_;
};

#endif
