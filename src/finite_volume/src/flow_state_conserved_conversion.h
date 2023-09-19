#ifndef FLOW_STATE_CONSERVED_CONVSERION_H
#define FLOW_STATE_CONSERVED_CONVSERION_H

#include "conserved_quantities.h"

template <typename T>
int conserved_to_flow_states(ConservedQuantities<T>& cq, FlowStates<T>& fs);

template <typename T>
int flow_states_to_conserved(ConservedQuantities<T>& cq, FlowStates<T>& fs);

#endif
