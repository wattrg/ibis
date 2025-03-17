#ifndef IBIS_PARALLEL_H
#define IBIS_PARALLEL_H

#include <parallel/distributed_memory.h>
#include <parallel/parallel_fwd.h>
#include <parallel/reductions.h>
#include <parallel/shared_memory.h>

namespace Ibis {

template <class MemModel>
void initialise(int argc, char** argv);

template <class MemModel>
void finalise();

template <class FunctorType>
KOKKOS_INLINE_FUNCTION void parallel_for(const std::string& str, const size_t work_count,
                                         FunctorType functor) {
    Kokkos::parallel_for(str, work_count, functor);
}

template <class FunctorType, class PolicyType>
KOKKOS_INLINE_FUNCTION void parallel_for(const std::string& str, PolicyType policy,
                                         FunctorType functor) {
    Kokkos::parallel_for(str, policy, functor);
}

template <class ReducerType, class MemModel, class FunctorType>
KOKKOS_INLINE_FUNCTION auto parallel_reduce(const std::string& str,
                                            const size_t work_count, FunctorType functor)
    -> typename ReducerType::scalar {
    using reducer_type = typename ReducerMap<ReducerType, MemModel>::reducer_type;
    reducer_type reducer;
    return reducer.execute(str, work_count, functor);
}

template <class ReducerType, class MemModel, class FunctorType, class PolicyType>
KOKKOS_INLINE_FUNCTION auto parallel_reduce(const std::string& str, PolicyType policy,
                                            FunctorType functor) ->
    typename ReducerType::scalar_type {
    using reducer_type = typename ReducerMap<ReducerType, MemModel>::reducer_type;
    reducer_type reducer;
    return reducer.execute(str, policy, functor);
}

}  // namespace Ibis

#endif
