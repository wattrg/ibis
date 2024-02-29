#include <finite_volume/flux_calc.h>
#include <gas/gas_model.h>
#include <gas/gas_state.h>
#include <pybind11/pybind11.h>
#include <util/vector3.h>

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <Kokkos_Core.hpp>

using ideal_gas = IdealGas<double>;

PYBIND11_MODULE(python_api, m) {
    m.doc() = "ibis python module";

    pybind11::class_<GasState<double>>(m, "GasState")
        .def(pybind11::init())
        .def_readwrite("p", &GasState<double>::pressure, "pressure (Pa)")
        .def_readwrite("T", &GasState<double>::temp, "temperature (K)")
        .def_readwrite("rho", &GasState<double>::rho, "density (kg/m^3)")
        .def_readwrite("energy", &GasState<double>::energy, "energy (J/kg)");

    pybind11::class_<ideal_gas>(m, "PyIdealGas")
        .def(pybind11::init<double>())
        .def("R", &ideal_gas::R)
        .def("Cv", &ideal_gas::Cv)
        .def("Cp", &ideal_gas::Cp)
        .def("gamma", &ideal_gas::gamma)
        .def("speed_of_sound",
             static_cast<double (ideal_gas::*)(const GasState<double>&) const>(
                 &ideal_gas::speed_of_sound))
        .def("update_thermo_from_pT",
             static_cast<void (ideal_gas::*)(GasState<double>&) const>(
                 &ideal_gas::update_thermo_from_pT))
        .def("update_thermo_from_rhoT",
             static_cast<void (ideal_gas::*)(GasState<double>&) const>(
                 &ideal_gas::update_thermo_from_rhoT))
        .def("update_thermo_from_rhop",
             static_cast<void (ideal_gas::*)(GasState<double>&) const>(
                 &ideal_gas::update_thermo_from_rhop));

    pybind11::class_<Vector3<double>>(m, "Vector3")
        .def(pybind11::init<>())
        .def(pybind11::init<double>())
        .def(pybind11::init<double, double>())
        .def(pybind11::init<double, double, double>())
        .def_readwrite("x", &Vector3<double>::x, "x")
        .def_readwrite("y", &Vector3<double>::y, "y")
        .def_readwrite("z", &Vector3<double>::z, "z");

    // flux calculators
    pybind11::class_<FluxCalculator<double>>(m, "FluxCalculator");

    pybind11::class_<Hanel<double>>(m, "PyHanel")
        .def(pybind11::init<>())
        .def("name", &Hanel<double>::name);

    pybind11::class_<Ausmdv<double>>(m, "PyAusmdv")
        .def(pybind11::init<>())
        .def("name", &Ausmdv<double>::name);

    pybind11::class_<Ldfss2<double>>(m, "PyLdfss2")
        .def(pybind11::init<>())
        .def("name", &Ldfss2<double>::name);
}
