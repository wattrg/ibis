#ifndef FLUX_H
#define FLUX_H

#include <gas/flow_state.h>
#include <gas/gas_model.h>
#include <grid/interface.h>
#include <finite_volume/conserved_quantities.h>

enum class FluxCalculator {
    Hanel,
    Ausmdv,
};

FluxCalculator flux_calculator_from_string(std::string name);
std::string string_from_flux_calculator(FluxCalculator flux_calc);

template <typename T>
void hanel(FlowStates<T>& left, FlowStates<T>& right,
           ConservedQuantities<T>& flux, IdealGas<T>& gm, bool three_d);

template <typename T>
void ausmdv(FlowStates<T>& left, FlowStates<T>& right,
            ConservedQuantities<T>& flux, IdealGas<T>& gm, bool three_d);

#endif
