#include <spdlog/spdlog.h>
#include <doctest/doctest.h>
#include "flux_calc.h"

FluxCalculator flux_calculator_from_string(std::string name){
    FluxCalculator flux_calc;
    if (name == "hanel") {
        flux_calc = FluxCalculator::Hanel;
    }
    else {
        spdlog::error("Unknown flux calculator {}", name);
        throw std::runtime_error("Unknown flux calculator");
    }
    return flux_calc;
}
