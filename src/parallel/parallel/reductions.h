#ifndef REDUCTIONS_H
#define REDUCTIONS_H

#include <Kokkos_Core.hpp>

template <typename T>
struct Min {
    using scalar_type = T;
};

template <typename T>
struct Max {
    using scalar_type = T;
};

template <typename T>
struct Sum {
    using scalar_type = T;
};

template <class Reduction>
struct ReductionIdentity;

template <typename Scalar>
struct ReductionIdentity<Min<Scalar>> {
    constexpr static Scalar identity() {
        return Kokkos::reduction_identity<Scalar>::min();
    }
};

template <typename Scalar>
struct ReductionIdentity<Max<Scalar>> {
    constexpr static Scalar identity() {
        return Kokkos::reduction_identity<Scalar>::max();
    }
};

template <typename Scalar>
struct ReductionIdentity<Sum<Scalar>> {
    constexpr static Scalar identity() {
        return Kokkos::reduction_identity<Scalar>::sum();
    }
};

namespace Ibis {}  // namespace Ibis

#endif
