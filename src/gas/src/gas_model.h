#ifndef GAS_MODEL_H
#define GAS_MODEL_H

#include <nlohmann/json.hpp>

#include "gas_state.h"

using json = nlohmann::json;

template <typename T>
class IdealGas {
public:
    IdealGas(double R);
    IdealGas(json config);

    // update an individual gas state
    void update_thermo_from_pT(GasState<T> &gs);
    void update_thermo_from_rhoT(GasState<T> &gs);
    void update_thermo_from_rhop(GasState<T> &gs);

    // update all the gas states
    void update_thermo_from_pT(GasStates<T> &gs);
    void update_thermo_from_rhoT(GasStates<T> &gs);
    void update_thermo_from_rhop(GasStates<T> &gs);

    // update a single gas state from the collection
    KOKKOS_INLINE_FUNCTION
    void update_thermo_from_pT(GasStates<T> &gs, int i);

    KOKKOS_INLINE_FUNCTION
    void update_thermo_from_rhoT(GasStates<T> &gs, int i);

    KOKKOS_INLINE_FUNCTION
    void update_thermo_from_rhop(GasStates<T> &gs, int i);

    // speed of sound
    T speed_of_sound(GasState<T> &gs);

    KOKKOS_INLINE_FUNCTION
    T speed_of_sound(GasStates<T> &gs, int i);

private:
    double R_;
    double Cv_;
    double Cp_;
    double gamma_;
};

#endif
