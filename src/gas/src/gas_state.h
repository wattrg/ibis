#ifndef GAS_H
#define GAS_H

#include "../../util/src/field.h"

template <typename T>
struct GasStates {
public:
    GasStates(int n) {
        _pressure = Aeolus::Field<T>("pressure", n);
        _temperature = Aeolus::Field<T>("temperature", n);
        _density = Aeolus::Field<T>("density", n);
        _energy = Aeolus::Field<T>("energy", n);
    }

private:
    Aeolus::Field<T> _pressure;
    Aeolus::Field<T> _temperature;
    Aeolus::Field<T> _density;
    Aeolus::Field<T> _energy;
};

#endif
