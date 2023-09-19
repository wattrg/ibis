#include "flow_state.h"

template <typename T>
FlowStates<T>::FlowStates(int n) : gas(GasStates<T>(n)), vel(Vector3s<T>(n)) {}

template <typename T>
void FlowStates<T>::copy_flow_state(const FlowState<T>& fs, const int i) {
    gas.copy_gas_state(fs.gas_state, i);
    vel.copy_vector(fs.velocity, i);
}

template class FlowStates<double>;
