#include "gas_model.h"

#include <Kokkos_Core.hpp>

template <typename T>
IdealGas<T>::IdealGas(double R) {
    R_ = R;
    Cv_ = 3.0 / 2.0 * R;
    Cp_ = 5.0 / 2.0 * R;
    gamma_ = Cp_ / Cv_;
}

template <typename T>
IdealGas<T>::IdealGas(json config)
    : R_(config.at("R")),
      Cv_(config.at("Cv")),
      Cp_(config.at("Cp")),
      gamma_(config.at("gamma")) {}

template <typename T>
KOKKOS_INLINE_FUNCTION T rho_from_pT(T p, T temp, double R) {
    return p / (R * temp);
}

template <typename T>
KOKKOS_INLINE_FUNCTION T T_from_rhop(T rho, T p, double R) {
    return p / (rho * R);
}

template <typename T>
KOKKOS_INLINE_FUNCTION T p_from_rhoT(T rho, T temp, double R) {
    return rho * R * temp;
}

template <typename T>
KOKKOS_INLINE_FUNCTION T speed_of_sound_(T temp, T R, T gamma) {
    return Kokkos::sqrt(gamma * R * temp);
}

template <typename T>
T IdealGas<T>::speed_of_sound(GasState<T> &gs) {
    return speed_of_sound_(gs.temp, R_, gamma_);
}

template <typename T>
T IdealGas<T>::speed_of_sound(GasStates<T> &gs, int i) {
    return Kokkos::sqrt(gamma_ * R_ * gs.temp(i));
}

template <typename T>
void IdealGas<T>::update_thermo_from_pT(GasState<T> &gs) {
    gs.rho = rho_from_pT(gs.pressure, gs.temp, R_);
}

template <typename T>
void IdealGas<T>::update_thermo_from_pT(GasStates<T> &gs, int i) {
    gs.rho(i) = rho_from_pT(gs.pressure(i), gs.temp(i), R_);
}

template <typename T>
void update_thermo_from_pT(GasStates<T> &gs) {
    Kokkos::parallel_for(
        "update_thermo_from_pT", gs.size(), KOKKOS_LAMBDA(const int i) {
            update_thermo_from_pT(gs, i);
        });
}

template <typename T>
void IdealGas<T>::update_thermo_from_rhoT(GasState<T> &gs) {
    gs.pressure = p_from_rhoT(gs.rho, gs.temp, R_);
}

template <typename T>
void IdealGas<T>::update_thermo_from_rhoT(GasStates<T> &gs, int i) {
    gs.pressure(i) = p_from_rhoT(gs.rho(i), gs.temp(i), R_);
}

template <typename T>
void update_thermo_from_rhoT(GasStates<T> &gs) {
    Kokkos::parallel_for(
        "update_thermo_from_rhoT", gs.size(), KOKKOS_LAMBDA(const int i) {
            update_thermo_from_rhoT(gs, i);
        });
}

template <typename T>
void IdealGas<T>::update_thermo_from_rhop(GasState<T> &gs) {
    gs.temp = T_from_rhop(gs.rho, gs.pressure, R_);
}

template <typename T>
void IdealGas<T>::update_thermo_from_rhop(GasStates<T> &gs, int i) {
    gs.temp(i) = T_from_rhop(gs.rho(i), gs.pressure(i), R_);
}

template <typename T>
void update_thermo_from_rhop(GasStates<T> &gs) {
    Kokkos::parallel_for(
        "update_thermo_from_rhop", gs.size(), KOKKOS_LAMBDA(const int i) {
            update_thermo_from_rhop(gs, i);
        });
}


template class IdealGas<double>;
