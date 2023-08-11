#ifndef FLOW_STATE_H
#define FLOW_STATE_H

#include "gas_state.h"
#include "../../util/src/vector3.h"

struct FlowStates{
public:
    FlowStates(int n) : gas(GasStates(n)), vel(Aeolus::Vector3s(n)) {}
    
    GasStates gas;
    Aeolus::Vector3s vel;
};

#endif
