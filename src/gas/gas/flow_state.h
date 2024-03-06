#ifndef FLOW_STATE_H
#define FLOW_STATE_H

#include <finite_volume/conserved_quantities.h>
#include <gas/gas_state.h>
#include <util/vector3.h>

#include <Kokkos_Core.hpp>

template <typename T>
struct FlowState {
public:
    FlowState() {}

    FlowState(GasState<T> gs, Vector3<T> vel) : gas_state(gs), velocity(vel) {}

    GasState<T> gas_state;
    Vector3<T> velocity;
};

template <typename T, class Layout = Kokkos::DefaultExecutionSpace::array_layout,
          class Space = Kokkos::DefaultExecutionSpace::memory_space>
struct FlowStates {
public:
    using array_layout = Layout;
    using memory_space = Space;
    using host_mirror_mem_space = Kokkos::DefaultHostExecutionSpace::memory_space;
    using host_mirror_layout = Kokkos::DefaultExecutionSpace::array_layout;
    using mirror_type = FlowStates<T, host_mirror_layout, host_mirror_mem_space>;

public:
    FlowStates() {}

    FlowStates(int n)
        : gas(GasStates<T, Layout, Space>(n)), vel(Vector3s<T, Layout, Space>(n)) {}

    FlowStates(GasStates<T, Layout, Space> gas, Vector3s<T, Layout, Space> vel)
        : gas(gas), vel(vel) {}

    int number_flow_states() const { return gas.size(); }

    mirror_type host_mirror() const {
        auto gas_mirror = gas.host_mirror();
        auto vel_mirror = vel.host_mirror();
        return mirror_type(gas_mirror, vel_mirror);
    }

    template <class OtherSpace>
    void deep_copy(const FlowStates<T, Layout, OtherSpace>& other) {
        gas.deep_copy(other.gas);
        vel.deep_copy(other.vel);
    }

    GasStates<T, Layout, Space> gas;
    Vector3s<T, Layout, Space> vel;
};

#endif
