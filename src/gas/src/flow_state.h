#ifndef FLOW_STATE_H
#define FLOW_STATE_H

#include "gas_state.h"
#include "../../util/src/vector3.h"

template <typename T>
struct FlowStates{
public:
    FlowStates(int n) 
        : gas(GasStates<T>(n)), vel(Vector3s<T>(n)) {}
    
    GasStates<T> gas;
    Vector3s<T> vel;
};

#endif
