#ifndef PRIMATIVE_CONSERVED_CONVSERION_H
#define PRIMATIVE_CONSERVED_CONVSERION_H

#include <util/conserved_quantities.h>
#include <gas/gas_model.h>
#include <gas/flow_state.h>

template <typename T>
int conserved_to_primatives(ConservedQuantities<T>& cq, FlowStates<T>& fs,
                            const IdealGas<T>& gm);

template <typename T>
int primatives_to_conserved(ConservedQuantities<T>& cq, FlowStates<T>& fs,
                            const IdealGas<T>& gm);

#endif
