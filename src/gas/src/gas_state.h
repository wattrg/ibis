#ifndef GAS_H
#define GAS_H

#include "Kokkos_Macros.hpp"
#include <Kokkos_Core.hpp>

template <typename T>
struct GasState {
public:
    GasState() {}

    T rho;
    T pressure;
    T temp;
    T energy;
};

template <typename T>
class GasStates {
public:
    GasStates(){}

    GasStates(int n) {
        rho_idx_ = 0;
        pressure_idx_ = 1;
        temp_idx_ = 2;
        energy_idx_ = 3;
        data_ = Kokkos::View<T**> ("GasStates", n, 4);
    }

    KOKKOS_INLINE_FUNCTION 
    T& rho(const int cell_i) const {return data_(cell_i, rho_idx_);}

    KOKKOS_INLINE_FUNCTION 
    T& rho(const int cell_i) {return data_(cell_i, rho_idx_);}

    KOKKOS_INLINE_FUNCTION
    T& pressure(const int cell_i) const {return data_(cell_i, pressure_idx_);}

    KOKKOS_INLINE_FUNCTION
    T& pressure(const int cell_i) {return data_(cell_i, pressure_idx_);}

    KOKKOS_INLINE_FUNCTION
    T& temp(const int cell_i) const {return data_(cell_i, temp_idx_);}

    KOKKOS_INLINE_FUNCTION
    T& temp(const int cell_i) {return data_(cell_i, temp_idx_);}

    KOKKOS_INLINE_FUNCTION
    T& energy(const int cell_i) const {return data_(cell_i, energy_idx_);}

    KOKKOS_INLINE_FUNCTION
    T& energy(const int cell_i) {return data_(cell_i, energy_idx_);}

    KOKKOS_INLINE_FUNCTION
    void copy_gas_state(const GasState<T>& gs, const int i) {
        rho(i) = gs.rho;
        pressure(i) = gs.pressure;
        temp(i) = gs.temp;
        energy(i) = gs.energy;
    }

    KOKKOS_INLINE_FUNCTION
    int size() const {return data_.extent(0);}

private:
    Kokkos::View<T**> data_;
    int rho_idx_, pressure_idx_, temp_idx_, energy_idx_;
};

#endif
