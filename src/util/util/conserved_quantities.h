#ifndef CONSERVED_QUANTITIES_H
#define CONSERVED_QUANTITIES_H

// #include <gas/flow_state.h>
#include <util/numeric_types.h>
#include <util/types.h>
// #include <ibis_mpi/ibis_mpi.h>
// #include <parallel/parallel.h>

#include <Kokkos_Core.hpp>
#include <fstream>

template <typename T>
class ConservedQuantitiesNorm {
public:
    KOKKOS_INLINE_FUNCTION
    ConservedQuantitiesNorm() {
        for (size_t i = 0; i < 6; i++) {
            data_[i] = T(0.0);
        }
    }

    KOKKOS_INLINE_FUNCTION
    ConservedQuantitiesNorm(const ConservedQuantitiesNorm<T>& rhs) {
        for (size_t i = 0; i < 6; i++) {
            data_[i] = rhs.data_[i];
        }
    }

    KOKKOS_INLINE_FUNCTION
    ConservedQuantitiesNorm& operator=(const ConservedQuantitiesNorm<T>& rhs) {
        for (size_t i = 0; i < 6; i++) {
            data_[i] = rhs.data_[i];
        }
        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    ConservedQuantitiesNorm& operator+=(const ConservedQuantitiesNorm<T>& rhs) {
        for (size_t i = 0; i < 6; i++) {
            data_[i] += rhs.data_[i];
        }
        return *this;
    }

    friend ConservedQuantitiesNorm operator/(const ConservedQuantitiesNorm<T> num,
                                             const ConservedQuantitiesNorm<T> den) {
        ConservedQuantitiesNorm<T> res{};
        for (size_t i = 0; i < 6; i++) {
            res.data_[i] = num.data_[i] / den.data_[i];
        }
        return res;
    }

    KOKKOS_INLINE_FUNCTION
    T& global() { return data_[0]; }

    KOKKOS_INLINE_FUNCTION
    T& mass() { return data_[1]; }

    KOKKOS_INLINE_FUNCTION
    T& momentum_x() { return data_[2]; }

    KOKKOS_INLINE_FUNCTION
    T& momentum_y() { return data_[3]; }

    KOKKOS_INLINE_FUNCTION
    T& momentum_z() { return data_[4]; }

    KOKKOS_INLINE_FUNCTION
    T& energy() { return data_[5]; }

    void write_to_file(std::ofstream& f, Ibis::real wc, Ibis::real time, size_t step);

private:
    T data_[6];
    // T global_;
    // T mass_;
    // T momentum_x_;
    // T momentum_y_;
    // T momentum_z_;
    // T energy_;
};

// this allows ConservedQuantity to be used as a custom scalar type
// for Kokkos reductions
namespace Kokkos {
template <typename T>
struct reduction_identity<ConservedQuantitiesNorm<T> > {
    KOKKOS_FORCEINLINE_FUNCTION
    static ConservedQuantitiesNorm<T> sum() { return ConservedQuantitiesNorm<T>(); }
};
}  // namespace Kokkos

#ifdef Ibis_ENABLE_MPI

// TODO: Custom MPI reductions for conserved quantity norms
#endif  // Ibis_ENABLE_MPI

template <typename T>
class ConservedQuantities {
public:
    ConservedQuantities() {}

    ConservedQuantities(size_t n, size_t dim);

    KOKKOS_INLINE_FUNCTION
    size_t size() const { return cq_.extent(0); }

    KOKKOS_INLINE_FUNCTION
    size_t n_conserved() const { return cq_.extent(1); }

    KOKKOS_INLINE_FUNCTION
    int dim() const { return dim_; }

    void apply_time_derivative(const ConservedQuantities<T>& dudt, Ibis::real dt);

    ConservedQuantitiesNorm<T> L2_norms() const;

    // ConservedQuantitiesNorm<Ibis::real> Linf_norms() const;

    KOKKOS_INLINE_FUNCTION
    T& operator()(const size_t cell_i, const size_t cq_i) { return cq_(cell_i, cq_i); }

    KOKKOS_INLINE_FUNCTION
    T& operator()(const size_t cell_i, const size_t cq_i) const {
        return cq_(cell_i, cq_i);
    }

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

    void deep_copy(const ConservedQuantities<T>& other);

private:
    Kokkos::View<T**> cq_;
    unsigned int mass_idx_, momentum_idx_, energy_idx_;
    int num_values_, dim_;
};

template <typename T>
void apply_time_derivative(const ConservedQuantities<T>& U0, ConservedQuantities<T>& U1,
                           ConservedQuantities<T>& dUdt, Ibis::real dt);

#endif
