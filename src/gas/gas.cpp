#include "gas.h"

GasStates::GasStates(int n) {
    _pressure = new double[n];
    _temp = new double[n];
}

GasStates::~GasStates() {
    delete[] _pressure;
    delete[] _temp;
}
