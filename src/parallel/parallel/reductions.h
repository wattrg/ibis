#ifndef REDUCTIONS_H
#define REDUCTIONS_H

#include <Kokkos_Core.hpp>
// #include <util/numeric_types.h>

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

// template <typename T>
// struct ReductionIdentity<Min<Ibis::Dual<T>>> {
//     constexpr static Ibis::Dual<T> identity() {
//         return Ibis::Dual<T> { ReductionIdentity<T>::identity(), T(0.0) };
//     }
// };

// template <typename T>
// struct ReductionIdentity<Max<Ibis::Dual<T>>> {
//     constexpr static Ibis::Dual<T> identity() {
//         return Ibis::Dual<T> { ReductionIdentity<T>::identity(), T(0.0) };
//     }
// };

// template <typename T>
// struct ReductionIdentity<Sum<Ibis::Dual<T>>> {
//     constexpr static Ibis::Dual<T> identity() {
//         return Ibis::Dual<T> { ReductionIdentity<T>::identity(), T(0.0) };
//     }
// };

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
