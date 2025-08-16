#ifndef IBIS_PARALLEL_H
#define IBIS_PARALLEL_H

#include <parallel/parallel_fwd.h>
#include <parallel/reductions.h>
#include <util/types.h>

namespace Ibis {

template <class MemModel>
void initialise(int argc, char** argv);

template <class MemModel>
void finalise();

template <class FunctorType>
inline void parallel_for(const std::string& str, const size_t work_count,
                         FunctorType functor) {
    Kokkos::parallel_for(str, work_count, functor);
}

template <class FunctorType, class PolicyType>
inline void parallel_for(const std::string& str, PolicyType policy, FunctorType functor) {
    Kokkos::parallel_for(str, policy, functor);
}

template <class ReducerType, class MemModel, class FunctorType>
inline auto parallel_reduce(const std::string& str, const size_t work_count,
                            FunctorType functor) -> typename ReducerType::scalar {
    using reducer_type = typename ReducerMap<ReducerType, MemModel>::reducer_type;
    reducer_type reducer;
    return reducer.execute(str, work_count, functor);
}

template <class ReducerType, class MemModel, class FunctorType, class PolicyType>
inline auto parallel_reduce(const std::string& str, PolicyType policy,
                            FunctorType functor) -> typename ReducerType::scalar_type {
    using reducer_type = typename ReducerMap<ReducerType, MemModel>::reducer_type;
    reducer_type reducer;
    return reducer.execute(str, policy, functor);
}

template <class MemModel, typename T, bool gpu_aware = false>
class SymmetricComm;

template <typename T, bool gpu_aware>
class SymmetricComm<SharedMem, T, gpu_aware> {
private:
    using MemSpace = Ibis::DefaultMemSpace;

public:
    SymmetricComm() {}
    SymmetricComm(int other_rank, size_t buf_size) {
        (void)other_rank;
        (void)buf_size;
        throw std::runtime_error("Not implemented");
    }
    SymmetricComm(int other_rank) {
        (void)other_rank;
        throw std::runtime_error("Not implemented");
    }
    SymmetricComm(const SymmetricComm& other) {
        (void)other;
        throw std::runtime_error("not implemented");
    }
    void expect_receive() { throw std::runtime_error("Not implemented"); }
    void send() { throw std::runtime_error("Not implemented"); }
    int receive() { throw std::runtime_error("Not implemented"); }
    void resize_buffers(size_t new_size) { throw std::runtime_error("Not implemented"); }
    const Kokkos::View<T*, MemSpace>& send_buf() const {
        throw std::runtime_error("Not implemented");
    }
    const Kokkos::View<T*, MemSpace>& recv_buf() const {
        throw std::runtime_error("Not implemented");
    }
};

}  // namespace Ibis

// Include the headers of the active memory models so that this
// header is the only thing that needs to be included in other files
#include <ibis_kokkos/ibis_kokkos.h>

#ifdef Ibis_ENABLE_MPI
#include <ibis_mpi/ibis_mpi.h>
#endif

#endif
