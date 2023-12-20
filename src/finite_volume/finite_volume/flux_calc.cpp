#include <doctest/doctest.h>
#include <finite_volume/flux_calc.h>
#include <spdlog/spdlog.h>

#include <stdexcept>

FluxCalculator flux_calculator_from_string(std::string name) {
    FluxCalculator flux_calc;
    if (name == "hanel") {
        flux_calc = FluxCalculator::Hanel;
    } else if (name == "ausmdv") {
        flux_calc = FluxCalculator::Ausmdv;
    } else {
        spdlog::error("Unknown flux calculator {}", name);
        throw std::runtime_error("Unknown flux calculator");
    }
    return flux_calc;
}

std::string string_from_flux_calculator(FluxCalculator flux_calc) {
    switch (flux_calc) {
        case FluxCalculator::Hanel:
            return "hanel";
        case FluxCalculator::Ausmdv:
            return "ausmdv";
        default:
            throw std::runtime_error("Shouldn't get here");
    }
}
