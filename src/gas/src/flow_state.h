#ifndef FLOW_STATE_H
#define FLOW_STATE_H

#include <Kokkos_Core.hpp>
#include "../../finite_volume/src/conserved_quantities.h"
#include "Kokkos_Core_fwd.hpp"
#include "gas_state.h"
#include "../../util/src/vector3.h"

template <typename T>
struct FlowState{
public:
    FlowState(){}

    // ~FlowState(){}

    FlowState(GasState<T> gs, Vector3<T> vel)
        : gas_state(gs), velocity(vel) {}
    

    GasState<T> gas_state;
    Vector3<T> velocity;
};

template <typename T,
          class Layout=Kokkos::DefaultExecutionSpace::array_layout,
          class Space=Kokkos::DefaultExecutionSpace::memory_space>
struct FlowStates{
public:
    using array_layout = Layout;
    using memory_space = Space;
    using host_mirror_memory_space = Kokkos::DefaultHostExecutionSpace::memory_space;
    using mirror_type = FlowStates<T, Layout, host_mirror_memory_space>;

public:
    FlowStates(){}

    FlowStates(int n) : 
        gas(GasStates<T, Layout, Space>(n)), 
        vel(Vector3s<T, Layout, Space>(n)){}

    int number_flow_states() const {return gas.size();}

    mirror_type host_mirror() {
        return mirror_type(number_flow_states());
    }

    template <class OtherSpace>
    void deep_copy(const FlowStates<T, Layout, OtherSpace>& other){
        gas.deep_copy(other.gas);
        vel.deep_copy(other.vel);
    }
    
    GasStates<T, Layout, Space> gas;
    Vector3s<T, Layout, Space> vel;
};

#endif
