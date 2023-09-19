#ifndef FLOW_STATE_H
#define FLOW_STATE_H

#include <Kokkos_Core.hpp>
#include "../../finite_volume/src/conserved_quantities.h"
#include "gas_state.h"
#include "../../util/src/vector3.h"

template <typename T>
struct FlowState{
public:
    FlowState(GasState<T> gs, Vector3<T> vel)
        : gas_state(gs), velocity(vel) {}
    

    GasState<T> gas_state;
    Vector3<T> velocity;
};

template <typename T>
struct FlowStates{
public:
    FlowStates(int n) 
        : gas(GasStates<T>(n)), vel(Vector3s<T>(n)) {}

    void copy_flow_state(const FlowState<T>& fs, const int i) {
        gas.copy_gas_state(fs.gas_state, i);
        vel.copy_vector(fs.velocity, i);
    }

    unsigned int number_flow_states() const {return gas.size();}

    
    GasStates<T> gas;
    Vector3s<T> vel;
};

#endif
