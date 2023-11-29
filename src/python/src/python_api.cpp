#include <pybind11/pybind11.h>

#include "../../finite_volume/src/flux_calc.h"
#include "../../gas/src/gas_state.h"
#include "../../gas/src/gas_model.h"
#include "../../util/src/vector3.h"

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

PYBIND11_MODULE(python_api, m) {
    m.doc() = "Aeolus python module";

    pybind11::class_<GasState<double>>(m, "GasState")
        .def(pybind11::init())
        .def_readwrite("p", &GasState<double>::pressure, "pressure (Pa)")
        .def_readwrite("T", &GasState<double>::temp, "temperature (K)")
        .def_readwrite("rho", &GasState<double>::rho, "density (kg/m^3)")
        .def_readwrite("a", &GasState<double>::a, "speed of sound (m/s)");

    pybind11::class_<IdealGas<double>>(m, "IdealGas")
        .def(pybind11::init<double>())
        .def("update_thermo_from_pT", 
             static_cast<void (IdealGas<double>::*)(GasState<double>&)>(&IdealGas<double>::update_thermo_from_pT));

    pybind11::class_<Vector3<double>>(m, "Vector3")
        .def(pybind11::init())
        .def_readwrite("x", &Vector3<double>::x, "x")
        .def_readwrite("y", &Vector3<double>::y, "y")
        .def_readwrite("z", &Vector3<double>::z, "z");

    pybind11::enum_<FluxCalculator>(m, "FluxCalculator")
        .value("Hanel", FluxCalculator::Hanel)
        .value("Ausmdv", FluxCalculator::Ausmdv);

    m.def("flux_calculator_from_string", &flux_calculator_from_string,
          "Convert string to flux calculator");

    m.def("string_from_flux_calculator", &string_from_flux_calculator,
          "Convert flux calculator to string");
}
