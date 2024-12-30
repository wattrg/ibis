#ifndef SHARED_MEMORY_H
#define SHARED_MEMORY_H

#include <Kokkos_Core.hpp>
#include <parallel/reductions.h>


template <class Scalar, class DistributedReducer>
struct SharedReducerMapping;

template <class Scalar>
struct SharedReducerMapping<Scalar, Min> {
    using value = Kokkos::Min<Scalar>;
};

template <class Scalar>
struct SharedReducerMapping<Scalar, Max> {
    using value = Kokkos::Max<Scalar>;
};

template <class Scalar>
struct SharedReducerMapping<Scalar, Sum> {
    using value = Kokkos::Sum<Scalar>;
};

#endif
