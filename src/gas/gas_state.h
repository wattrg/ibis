#ifndef GAS_H
#define GAS_H

#include "../util/field.h"

struct GasStates {
public:
    GasStates(int n);

private:
    Aeolus::Field _pressure;
    Aeolus::Field _temperature;
    Aeolus::Field _density;
};

#endif
