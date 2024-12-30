#ifndef DISTRIBUTED_H
#define DISTRIBUTED_H

#include <Kokkos_Core.hpp>
#include <parallel/reductions.h>
#include <parallel/mpi/ibis_mpi.h>
#include <parallel/shared_memory.h>


namespace Ibis {
namespace Distributed{

template <typename Scalar, class Reduction>
struct ReductionIdentity;

template <typename Scalar>
struct ReductionIdentity<Scalar, Min> {
    constexpr static Scalar identity() {
        return Kokkos::reduction_identity<Scalar>::min();
    }  
};

template <typename Scalar>
struct ReductionIdentity<Scalar, Max> {
    constexpr static Scalar identity() {
        return Kokkos::reduction_identity<Scalar>::max();
    }  
};

template <typename Scalar>
struct ReductionIdentity<Scalar, Sum> {
    constexpr static Scalar identity() {
        return Kokkos::reduction_identity<Scalar>::sum();
    }  
};


template <class FunctorType, class Reducer>
inline auto parallel_reduce(const std::string& str, const size_t work_count,
                           FunctorType functor,
                           Reducer reducer) -> typename Reducer::scalar_type {
    using scalar_type = typename Reducer::scalar_type;
    using reduction = typename Reducer::reduction;
    using shared_reducer = typename SharedReducerMapping<scalar_type, reduction>::value;
    scalar_type local_reduction = ReductionIdentity<scalar_type, reduction>::identity();
    shared_reducer local_reducer(local_reduction);
    Kokkos::parallel_reduce(str, work_count, functor, local_reducer);
    return reducer.reduce(local_reduction);
}


    
}
}

#endif
