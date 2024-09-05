#ifndef PRIMATIVE_CONSERVED_CONVSERION_H
#define PRIMATIVE_CONSERVED_CONVSERION_H

#include <finite_volume/conserved_quantities.h>
#include <gas/gas_model.h>

template <typename T>
int conserved_to_primitives(ConservedQuantities<T>& cq, FlowStates<T>& fs,
                            const IdealGas<T>& gm);

template <typename T>
int primitives_to_conserved(ConservedQuantities<T>& cq, FlowStates<T>& fs,
                            const IdealGas<T>& gm);

#endif
