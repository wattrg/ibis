#include "gas_state.h"

GasStates::GasStates(int n) {
    _pressure = Aeolus::Field("pressure", n);
    _temperature = Aeolus::Field("temperature", n);
    _density = Aeolus::Field("density", n);
}
