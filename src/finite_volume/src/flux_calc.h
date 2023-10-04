#ifndef FLUX_H
#define FLUX_H

#include "../../gas/src/flow_state.h"
#include "../../grid/src/interface.h"
#include "conserved_quantities.h"


enum class FluxCalculator {
    Hanel,
};

FluxCalculator flux_calculator_from_string(std::string name);


template <typename T>
void hanel(FlowStates<T>& left, FlowStates<T>& right, ConservedQuantities<T>& flux, bool three_d);

#endif
