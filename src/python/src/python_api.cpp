#include <pybind11/pybind11.h>
#include "../../gas/src/gas_state.h"
#include "../../util/src/vector3.h"

PYBIND11_MODULE(python_api, m) {
    m.doc() = "Aeolus python module";

    pybind11::class_<GasState<double>>(m, "GasState")
        .def(pybind11::init())
        .def_readwrite("p", &GasState<double>::pressure, "pressure (Pa)")
        .def_readwrite("T", &GasState<double>::temp, "temperature (K)")
        .def_readwrite("rho", &GasState<double>::rho, "density (kg/m^3)");

    pybind11::class_<Vector3<double>>(m, "Vector3")
        .def(pybind11::init())
        .def_readwrite("x", &Vector3<double>::x, "x")
        .def_readwrite("y", &Vector3<double>::y, "y")
        .def_readwrite("z", &Vector3<double>::z, "z");
}
