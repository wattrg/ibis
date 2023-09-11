#ifndef FV_IO_H
#define FV_IO_H

#include <string>
#include "../../gas/src/flow_state.h"

enum class FlowFormat {
    Native, Vtk
};

template <typename T>
class FVIO {
public:
    FVIO(FlowFormat format, int time_index, int block_index) 
        : format_(format), 
          time_index_(time_index), 
          block_index_(block_index) {}

    FVIO(FlowFormat format) : FVIO(format, 0, 0) {}

    FVIO() : FVIO(FlowFormat::Native, 0, 0) {}

    int write(FlowStates<T>& flow_state, double time);
    int write(FlowStates<T>& flow_state);

private:
    FlowFormat format_;
    int time_index_ = 0;
    int block_index_ = 0;
    std::string directory_;
};

#endif
