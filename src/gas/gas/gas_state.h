#ifndef GAS_H
#define GAS_H

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

template <typename T,
          class Layout = Kokkos::DefaultExecutionSpace::array_layout,
          class Space = Kokkos::DefaultExecutionSpace::memory_space>
class GasStates {
public:
    using view_type = Kokkos::View<T**, Layout, Space>;
    using array_layout = Layout;
    using memory_space = Space;
    using mirror_view_type = typename view_type::host_mirror_type;
    using mirror_layout = typename mirror_view_type::array_layout;
    using mirror_space = typename mirror_view_type::memory_space;
    using mirror_type = GasStates<T, mirror_layout, mirror_space>;

public:
    GasStates() {}

    GasStates(int n) {
        rho_idx_ = 0;
        pressure_idx_ = 1;
        temp_idx_ = 2;
        energy_idx_ = 3;
        data_ = view_type("GasStates", n, 4);
    }

    KOKKOS_INLINE_FUNCTION
    T& rho(const int cell_i) const { return data_(cell_i, rho_idx_); }

    KOKKOS_INLINE_FUNCTION
    T& rho(const int cell_i) { return data_(cell_i, rho_idx_); }

    KOKKOS_INLINE_FUNCTION
    T& pressure(const int cell_i) const { return data_(cell_i, pressure_idx_); }

    KOKKOS_INLINE_FUNCTION
    T& pressure(const int cell_i) { return data_(cell_i, pressure_idx_); }

    KOKKOS_INLINE_FUNCTION
    T& temp(const int cell_i) const { return data_(cell_i, temp_idx_); }

    KOKKOS_INLINE_FUNCTION
    T& temp(const int cell_i) { return data_(cell_i, temp_idx_); }

    KOKKOS_INLINE_FUNCTION
    T& energy(const int cell_i) const { return data_(cell_i, energy_idx_); }

    KOKKOS_INLINE_FUNCTION
    T& energy(const int cell_i) { return data_(cell_i, energy_idx_); }

    KOKKOS_INLINE_FUNCTION
    void copy_gas_state(const GasState<T>& gs, const int i) {
        rho(i) = gs.rho;
        pressure(i) = gs.pressure;
        temp(i) = gs.temp;
        energy(i) = gs.energy;
    }

    mirror_type host_mirror() const { return mirror_type(size()); }

    template <class OtherSpace>
    void deep_copy(const GasStates<T, Layout, OtherSpace>& other) {
        Kokkos::deep_copy(data_, other.data_);
    }

    KOKKOS_INLINE_FUNCTION
    int size() const { return data_.extent(0); }

public:
    view_type data_;
    int rho_idx_, pressure_idx_, temp_idx_, energy_idx_;
};

#endif
