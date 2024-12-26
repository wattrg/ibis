#ifndef DISTRIBUTED_H
#define DISTRIBUTED_H

#include <Kokkos_Core.hpp>
#include <parallel/mpi/ibis_mpi.h>

namespace Ibis {
namespace Distributed{

template <class FunctorType, class Reducer>
auto parallel_reduce(const std::string& str, const size_t work_count,
                     FunctorType functor, Reducer reducer) -> typename Reducer::scalar_type {
    typename Reducer::scalar_type local_reduction;
    typename Reducer::shared_reducer local_reducer(local_reduction);
    Kokkos::parallel_reduce(str, work_count, functor, local_reducer);
    return reducer.reduce(local_reduction);    
}

inline void parallel_reduce(const size_t work_count) {
    parallel_reduce(work_count);
}

    
}
}

#endif
