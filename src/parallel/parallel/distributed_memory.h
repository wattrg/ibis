#ifndef DISTRIBUTED_H
#define DISTRIBUTED_H

#include <parallel/reductions.h>
#include <parallel/shared_memory.h>

#include <Kokkos_Core.hpp>

#ifdef Ibis_ENABLE_MPI
#include <mpi/ibis_mpi.h>
#endif

namespace Ibis {

template <class SharedPolicy, class MemModel>
struct Policy {
    using shared_policy = SharedPolicy;
    using mem_model = MemModel;
};

template <class ReducerType, class FunctorType>
inline auto parallel_reduce(const std::string& str, const size_t work_count,
                            FunctorType functor) -> typename ReducerType::scalar_type {
    using scalar_type = typename ReducerType::scalar_type;

    // The shared memory part of the reduction
    scalar_type local_reduction =
        Ibis::Shared::parallel_reduce<ReducerType>(str, work_count, functor);

    // The distributed part of the reduction
    Distributed::DistributedReduction<ReducerType, DefaultMemModel> reducer;
    return reducer.reduce(local_reduction);
}

template <class ReducerType, class FunctorType, class PolicyType>
inline auto parallel_reduce(const std::string& str, PolicyType policy,
                            FunctorType functor) -> typename ReducerType::scalar_type {
    using scalar_type = typename ReducerType::scalar_type;

    // The shared memory part of the reduction
    scalar_type local_reduction =
        Ibis::Shared::parallel_reduce<ReducerType>(str, policy, functor);

    // The distributed part of the reduction
    Distributed::DistributedReduction<ReducerType, DefaultMemModel> reducer;
    return reducer.reduce(local_reduction);
}

}  // namespace Ibis

#endif
