#ifndef GAS_MODEL_H
#define GAS_MODEL_H

#include <nlohmann/json.hpp>

#include "Kokkos_Core_fwd.hpp"
#include "gas_state.h"

using json = nlohmann::json;

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

template <typename T,
          class Layout = Kokkos::DefaultExecutionSpace::array_layout,
          class Space = Kokkos::DefaultExecutionSpace::memory_space>
class IdealGas {
public:
    using layout = Layout;
    using space = Space;
    using host_mirror_mem_space =
        Kokkos::DefaultHostExecutionSpace::memory_space;
    using host_mirror_layout = Kokkos::DefaultExecutionSpace::array_layout;
    using mirror_type = IdealGas<T, host_mirror_layout, host_mirror_mem_space>;

public:
    IdealGas() {}

    IdealGas(double R) {
        R_ = R;
        Cv_ = 5.0 / 2.0 * R;
        Cp_ = 7.0 / 2.0 * R;
        gamma_ = Cp_ / Cv_;
    }

    IdealGas(json config)
        : R_(config.at("R")),
          Cv_(config.at("Cv")),
          Cp_(config.at("Cp")),
          gamma_(config.at("gamma")) {}

    // update an individual gas state
    void update_thermo_from_pT(GasState<T> &gs) const {
        gs.rho = rho_from_pT(gs.pressure, gs.temp, R_);
    }

    void update_thermo_from_rhoT(GasState<T> &gs) const {
        gs.pressure = p_from_rhoT(gs.rho, gs.temp, R_);
    }

    void update_thermo_from_rhop(GasState<T> &gs) const {
        gs.temp = T_from_rhop(gs.rho, gs.pressure, R_);
    }

    // update a single gas state from the collection
    KOKKOS_INLINE_FUNCTION
    void update_thermo_from_pT(GasStates<T, layout, space> &gs,
                               const int i) const {
        gs.rho(i) = rho_from_pT(gs.pressure(i), gs.temp(i), R_);
    }

    KOKKOS_INLINE_FUNCTION
    void update_thermo_from_rhoT(GasStates<T, layout, space> &gs,
                                 const int i) const {
        gs.pressure(i) = p_from_rhoT(gs.rho(i), gs.temp(i), R_);
    }

    KOKKOS_INLINE_FUNCTION
    void update_thermo_from_rhop(GasStates<T, layout, space> &gs,
                                 const int i) const {
        gs.temp(i) = T_from_rhop(gs.rho(i), gs.pressure(i), R_);
    }

    // update all the gas states
    void update_thermo_from_pT(GasStates<T, layout, space> &gs) const {
        Kokkos::parallel_for(
            "update_thermo_from_pT", gs.size(),
            KOKKOS_CLASS_LAMBDA(const int i) { update_thermo_from_pT(gs, i); });
    }
    void update_thermo_from_rhoT(GasStates<T, layout, space> &gs) const {
        Kokkos::parallel_for(
            "update_thermo_from_rhoT", gs.size(),
            KOKKOS_CLASS_LAMBDA(const int i) {
                update_thermo_from_rhoT(gs, i);
            });
    }

    void update_thermo_from_rhop(GasStates<T, layout, space> &gs) const {
        Kokkos::parallel_for(
            "update_thermo_from_rhop", gs.size(),
            KOKKOS_CLASS_LAMBDA(const int i) {
                update_thermo_from_rhop(gs, i);
            });
    }

    // speed of sound
    T speed_of_sound(const GasState<T> &gs) const {
        return speed_of_sound_(gs.temp, R_, gamma_);
    }

    KOKKOS_INLINE_FUNCTION
    T speed_of_sound(const GasStates<T> &gs, const int i) const {
        return speed_of_sound_(gs.temp(i), R_, gamma_);
    }

    double R() { return R_; }
    double Cv() { return Cv_; }
    double Cp() { return Cp_; }
    double gamma() { return gamma_; }

private:
    double R_;
    double Cv_;
    double Cp_;
    double gamma_;
};

#endif
