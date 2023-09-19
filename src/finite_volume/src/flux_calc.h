#ifndef FLUX_H
#define FLUX_H

#include "../../gas/src/flow_state.h"
#include "../../grid/src/interface.h"
#include "conserved_quantities.h"

#include "flux_calculators/hanel.h"

enum class FluxCalculator {
    Hanel,
};

template <typename T>
void compute_flux(FlowStates<T>& left, FlowStates<T>& right, 
                  ConservedQuantities<T>& flux, Interfaces<T>& faces, 
                  FluxCalculator flux_calc, bool three_d);

#endif
