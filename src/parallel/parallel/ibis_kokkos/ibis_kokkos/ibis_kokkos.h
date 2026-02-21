#ifndef SHARED_MEMORY_H
#define SHARED_MEMORY_H

#include <parallel/parallel_fwd.h>
#include <parallel/reductions.h>

#include <Kokkos_Core.hpp>

// Convert the Ibis reductions to Kokkos reductions
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

// Reductions
template <class ReducerType>
class Reducer<ReducerType, SharedMem> {
public:
    template <class FunctorType>
    inline auto execute(const std::string& str, const size_t work_count,
                        FunctorType functor) -> typename ReducerType::scalar_type {
        using scalar_type = typename ReducerType::scalar_type;
        using reduction = ReducerType;
        using shared_reducer = typename SharedReducerMapping<reduction>::value;
        scalar_type local_reduction;
        shared_reducer local_reducer(local_reduction);
        Kokkos::parallel_reduce(str, work_count, functor, local_reducer);
        return local_reduction;
    }

    template <class PolicyType, class FunctorType>
    inline auto execute(const std::string& str, PolicyType& policy, FunctorType functor)
        -> typename ReducerType::scalar_type {
        using scalar_type = typename ReducerType::scalar_type;
        using reduction = ReducerType;
        using shared_reducer = typename SharedReducerMapping<reduction>::value;
        scalar_type local_reduction;
        shared_reducer local_reducer(local_reduction);
        Kokkos::parallel_reduce(str, policy, functor, local_reducer);
        return local_reduction;
    }
};

template <class ReducerType>
struct ReducerMap<ReducerType, SharedMem> {
    using reducer_type = Reducer<ReducerType, SharedMem>;
};

}  // namespace Ibis

#endif
