#ifndef CONSERVED_QUANTITIES_H
#define CONSERVED_QUANTITIES_H

#include <Kokkos_Core.hpp>

#include "../../gas/src/flow_state.h"
#include "Kokkos_Macros.hpp"

template <typename T>
class ConservedQuantities {
   public:
    ConservedQuantities() {}

    ConservedQuantities(unsigned int n, unsigned int dim);

    KOKKOS_INLINE_FUNCTION
    unsigned int size() const { return cq_.extent(0); }

    KOKKOS_INLINE_FUNCTION
    unsigned int n_conserved() const { return cq_.extent(1); }

    KOKKOS_INLINE_FUNCTION
    int dim() const { return dim_; }

    void apply_time_derivative(const ConservedQuantities<T>& dudt, double dt);

    KOKKOS_INLINE_FUNCTION
    T& mass(int cell_i) const { return cq_(cell_i, mass_idx_); }

    KOKKOS_INLINE_FUNCTION
    T& mass(int cell_i) { return cq_(cell_i, mass_idx_); }

    KOKKOS_INLINE_FUNCTION
    T& momentum_x(int cell_i) const { return cq_(cell_i, momentum_idx_ + 0); }

    KOKKOS_INLINE_FUNCTION
    T& momentum_x(int cell_i) { return cq_(cell_i, momentum_idx_ + 0); }

    KOKKOS_INLINE_FUNCTION
    T& momentum_y(int cell_i) const { return cq_(cell_i, momentum_idx_ + 1); }

    KOKKOS_INLINE_FUNCTION
    T& momentum_y(int cell_i) { return cq_(cell_i, momentum_idx_ + 1); }

    KOKKOS_INLINE_FUNCTION
    T& momentum_z(int cell_i) const {
        assert(dim_ == 3);
        return cq_(cell_i, momentum_idx_ + 2);
    }

    KOKKOS_INLINE_FUNCTION
    T& momentum_z(int cell_i) {
        assert(dim_ == 3);
        return cq_(cell_i, momentum_idx_ + 2);
    }

    KOKKOS_INLINE_FUNCTION
    T& energy(int cell_i) const { return cq_(cell_i, energy_idx_); }

    KOKKOS_INLINE_FUNCTION
    T& energy(int cell_i) { return cq_(cell_i, energy_idx_); }

   private:
    Kokkos::View<T**> cq_;
    unsigned int mass_idx_, momentum_idx_, energy_idx_;
    int num_values_, dim_;
};

#endif
