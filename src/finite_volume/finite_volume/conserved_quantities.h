#ifndef CONSERVED_QUANTITIES_H
#define CONSERVED_QUANTITIES_H

#include <gas/flow_state.h>
#include <util/types.h>

#include <Kokkos_Core.hpp>
#include <fstream>

template <typename T>
class ConservedQuantitiesNorm {
public:
    KOKKOS_INLINE_FUNCTION
    ConservedQuantitiesNorm() {
        mass_ = 0.0;
        momentum_x_ = 0.0;
        momentum_y_ = 0.0;
        momentum_z_ = 0.0;
        energy_ = 0.0;
    }

    KOKKOS_INLINE_FUNCTION
    ConservedQuantitiesNorm(const ConservedQuantitiesNorm<T>& rhs) {
        mass_ = rhs.mass_;
        momentum_x_ = rhs.momentum_x_;
        momentum_y_ = rhs.momentum_y_;
        momentum_z_ = rhs.momentum_z_;
        energy_ = rhs.energy_;
    }

    KOKKOS_INLINE_FUNCTION
    ConservedQuantitiesNorm& operator=(const ConservedQuantitiesNorm<T>& rhs) {
        mass_ = rhs.mass_;
        momentum_x_ = rhs.momentum_x_;
        momentum_y_ = rhs.momentum_y_;
        momentum_z_ = rhs.momentum_z_;
        energy_ = rhs.energy_;
        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    ConservedQuantitiesNorm& operator+=(const ConservedQuantitiesNorm<T>& rhs) {
        mass_ += rhs.mass_;
        momentum_x_ += rhs.momentum_x_;
        momentum_y_ += rhs.momentum_y_;
        momentum_z_ += rhs.momentum_z_;
        energy_ += rhs.energy_;
        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    T& mass() { return mass_; }

    KOKKOS_INLINE_FUNCTION
    T& momentum_x() { return momentum_x_; }

    KOKKOS_INLINE_FUNCTION
    T& momentum_y() { return momentum_y_; }

    KOKKOS_INLINE_FUNCTION
    T& momentum_z() { return momentum_z_; }

    KOKKOS_INLINE_FUNCTION
    T& energy() { return energy_; }

    void write_to_file(std::ofstream& f, double time, unsigned int step);

private:
    T mass_;
    T momentum_x_;
    T momentum_y_;
    T momentum_z_;
    T energy_;
};

// this allows ConservedQuantity to be used as a custom scalar type
// for Kokkos reductions
namespace Kokkos {
template <>
struct reduction_identity<ConservedQuantitiesNorm<double> > {
    KOKKOS_FORCEINLINE_FUNCTION
    static ConservedQuantitiesNorm<double> sum() {
        return ConservedQuantitiesNorm<double>();
    }
};
}  // namespace Kokkos

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

    ConservedQuantitiesNorm<double> L2_norms() const;

    // ConservedQuantitiesNorm<double> Linf_norms() const;

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

template <typename T>
void apply_time_derivative(const ConservedQuantities<T>& U0, ConservedQuantities<T>& U1,
                           ConservedQuantities<T>& dUdt, double dt);

#endif
