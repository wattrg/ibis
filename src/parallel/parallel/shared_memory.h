#ifndef SHARED_MEMORY_H
#define SHARED_MEMORY_H

#include <parallel/reductions.h>

#include <Kokkos_Core.hpp>

template <class Reduction>
struct SharedReducerMapping;

template <class Scalar>
struct SharedReducerMapping<Min<Scalar>> {
    using value = Kokkos::Min<Scalar>;
};

template <class Scalar>
struct SharedReducerMapping<Max<Scalar>> {
    using value = Kokkos::Max<Scalar>;
};

template <class Scalar>
struct SharedReducerMapping<Sum<Scalar>> {
    using value = Kokkos::Sum<Scalar>;
};

namespace Ibis {
namespace Shared {

template <class ReducerType, class FunctorType>
inline auto parallel_reduce(const std::string& str, const size_t work_count,
                            FunctorType functor) -> typename ReducerType::scalar_type {
    using scalar_type = typename ReducerType::scalar_type;
    using reduction = ReducerType;
    using shared_reducer = typename SharedReducerMapping<reduction>::value;
    scalar_type local_reduction;
    shared_reducer local_reducer(local_reduction);
    Kokkos::parallel_reduce(str, work_count, functor, local_reducer);
    return local_reduction;
}

template <class ReducerType, class FunctorType, class PolicyType>
inline auto parallel_reduce(const std::string& str, PolicyType& policy,
                            FunctorType functor) -> typename ReducerType::scalar_type {
    using scalar_type = typename ReducerType::scalar_type;
    using reduction = ReducerType;
    using shared_reducer = typename SharedReducerMapping<reduction>::value;
    scalar_type local_reduction;
    shared_reducer local_reducer(local_reduction);
    Kokkos::parallel_reduce(str, policy, functor, local_reducer);
    return local_reduction;
}

}  // namespace Shared
}  // namespace Ibis

#endif
