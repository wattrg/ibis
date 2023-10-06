#ifndef PRIMATIVE_CONSERVED_CONVSERION_H
#define PRIMATIVE_CONSERVED_CONVSERION_H

#include "conserved_quantities.h"

template <typename T>
int conserved_to_primatives(ConservedQuantities<T>& cq, FlowStates<T>& fs);

template <typename T>
int primatives_to_conserved(ConservedQuantities<T>& cq, FlowStates<T>& fs);

#endif
