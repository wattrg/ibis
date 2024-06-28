#ifndef GAS_H
#define GAS_H

#include <util/types.h>

#include <Kokkos_Core.hpp>

#include "Kokkos_Macros.hpp"

template <typename T>
struct GasState {
public:
    KOKKOS_INLINE_FUNCTION
    GasState() {}

    T rho;
    T pressure;
    T temp;
    T energy;
};

template <typename T, class Layout = Kokkos::DefaultExecutionSpace::array_layout,
          class Space = Kokkos::DefaultExecutionSpace::memory_space>
class GasStates {
private:
    // using view_type = Kokkos::View<T**, Layout, Space>;
    using view_type = Ibis::Array2D<T, Layout, Space>;

public:
    using array_layout = Layout;
    using memory_space = Space;
    using mirror_view_type = typename view_type::host_mirror_type;
    using mirror_layout = typename mirror_view_type::array_layout;
    using mirror_space = typename mirror_view_type::memory_space;
    using mirror_type = GasStates<T, mirror_layout, mirror_space>;

public:
    GasStates() {}

    GasStates(size_t n) {
        rho_idx_ = 0;
        pressure_idx_ = 1;
        temp_idx_ = 2;
        energy_idx_ = 3;
        data_ = view_type("GasStates", n, 4);
    }

    GasStates(view_type gas_data)
        : data_(gas_data), rho_idx_(0), pressure_idx_(1), temp_idx_(2), energy_idx_(3) {}

    KOKKOS_INLINE_FUNCTION
    T& rho(const size_t cell_i) const { return data_(cell_i, rho_idx_); }

    KOKKOS_INLINE_FUNCTION
    T& rho(const size_t cell_i) { return data_(cell_i, rho_idx_); }

    KOKKOS_INLINE_FUNCTION
    auto rho() { return Kokkos::subview(data_, Kokkos::ALL, rho_idx_); }

    KOKKOS_INLINE_FUNCTION
    auto rho() const { return Kokkos::subview(data_, Kokkos::ALL, rho_idx_); }

    KOKKOS_INLINE_FUNCTION
    T& pressure(const size_t cell_i) const { return data_(cell_i, pressure_idx_); }

    KOKKOS_INLINE_FUNCTION
    T& pressure(const size_t cell_i) { return data_(cell_i, pressure_idx_); }

    KOKKOS_INLINE_FUNCTION
    auto pressure() { return Kokkos::subview(data_, Kokkos::ALL, pressure_idx_); }

    KOKKOS_INLINE_FUNCTION
    auto pressure() const { return Kokkos::subview(data_, Kokkos::ALL, pressure_idx_); }

    KOKKOS_INLINE_FUNCTION
    T& temp(const size_t cell_i) const { return data_(cell_i, temp_idx_); }

    KOKKOS_INLINE_FUNCTION
    T& temp(const size_t cell_i) { return data_(cell_i, temp_idx_); }

    KOKKOS_INLINE_FUNCTION
    auto temp() { return Kokkos::subview(data_, Kokkos::ALL, temp_idx_); }

    KOKKOS_INLINE_FUNCTION
    auto temp() const { return Kokkos::subview(data_, Kokkos::ALL, temp_idx_); }

    KOKKOS_INLINE_FUNCTION
    T& energy(const size_t cell_i) const { return data_(cell_i, energy_idx_); }

    KOKKOS_INLINE_FUNCTION
    T& energy(const size_t cell_i) { return data_(cell_i, energy_idx_); }

    KOKKOS_INLINE_FUNCTION
    auto energy() { return Kokkos::subview(data_, Kokkos::ALL, energy_idx_); }

    KOKKOS_INLINE_FUNCTION
    auto energy() const { return Kokkos::subview(data_, Kokkos::ALL, energy_idx_); }

    KOKKOS_INLINE_FUNCTION
    void set_gas_state(const GasState<T>& gs, const size_t i) const {
        rho(i) = gs.rho;
        pressure(i) = gs.pressure;
        temp(i) = gs.temp;
        energy(i) = gs.energy;
    }

    mirror_type host_mirror() const {
        mirror_view_type data = Kokkos::create_mirror_view(data_);
        return mirror_type(data);
    }

    template <class OtherSpace>
    void deep_copy(const GasStates<T, Layout, OtherSpace>& other) {
        Kokkos::deep_copy(data_, other.data_);
    }

    KOKKOS_INLINE_FUNCTION
    size_t size() const { return data_.extent(0); }

public:
    view_type data_;
    int rho_idx_, pressure_idx_, temp_idx_, energy_idx_;
};

#endif
