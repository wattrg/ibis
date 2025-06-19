#ifndef IBIS_MPI_H
#define IBIS_MPI_H

#ifdef Ibis_ENABLE_MPI

#include <mpi.h>
#include <parallel/parallel.h>
#include <ibis_kokkos/ibis_kokkos.h>
#include <util/types.h>
#include <util/numeric_types.h>
#include <util/conserved_quantities.h>


namespace Ibis {

// MPI data types
template <typename Type>
struct MpiDataType;

template <typename Type>
struct MpiDataType;

#define MpiTypeMapping(type, MPI_type)                   \
    template <>                                          \
    struct MpiDataType<type> {                           \
        static MPI_Datatype value() { return MPI_type; } \
    };

template <>
struct MpiDataType<Dual<double>> {
    static MPI_Datatype value() { return MPI_2DOUBLE_PRECISION; }  
};

template <>
struct MpiDataType<Dual<float>> {
    static MPI_Datatype value() { return MPI_2REAL; }  
};

MpiTypeMapping(short int, MPI_SHORT)                                // NOLINT
    MpiTypeMapping(int, MPI_INT)                                    // NOLINT
    MpiTypeMapping(long int, MPI_LONG)                              // NOLINT
    MpiTypeMapping(long long int, MPI_LONG_LONG)                    // NOLINT
    MpiTypeMapping(unsigned char, MPI_UNSIGNED_CHAR)                // NOLINT
    MpiTypeMapping(unsigned short int, MPI_UNSIGNED_SHORT)          // NOLINT
    MpiTypeMapping(unsigned int, MPI_UNSIGNED)                      // NOLINT
    MpiTypeMapping(unsigned long int, MPI_UNSIGNED_LONG)            // NOLINT
    MpiTypeMapping(unsigned long long int, MPI_UNSIGNED_LONG_LONG)  // NOLINT
    MpiTypeMapping(float, MPI_FLOAT)                                // NOLINT
    MpiTypeMapping(double, MPI_DOUBLE)                              // NOLINT
    MpiTypeMapping(long double, MPI_LONG_DOUBLE)                    // NOLINT
    MpiTypeMapping(char, MPI_CHAR)                                  // NOLINT

// Custom MPI operations of standard overloaded operators
template <typename T>
void MPI_custom_sum(T* invec, T* inoutvec, int* len, MPI_Datatype* datatype) {
    (void) datatype;
    for(int i = 0; i < *len; i++) {
        inoutvec[i] += invec[i];
    }
}

template <typename T>
void MPI_custom_max(T* invec, T* inoutvec, int* len, MPI_Datatype* datatype) {
    (void) datatype;
    for(int i = 0; i < *len; i++) {
        inoutvec[i] = max(invec[i], inoutvec[i]);
    }
}

template <typename T>
void MPI_custom_min(T* invec, T* inoutvec, int* len, MPI_Datatype* datatype) {
    (void) datatype;
    for(int i = 0; i < *len; i++) {
        inoutvec[i] = min(invec[i], inoutvec[i]);
    }
}
    
 
// Map between Ibis reduction types and MPI reduction types
template <typename Reduction>
struct MpiReduction;

template <typename T>
struct MpiReduction<Min<T>> {
    static MPI_Op op() { return MPI_MIN; }
};

template <typename T>
struct MpiReduction<Max<T>> {
    static MPI_Op op() { return MPI_MAX; }
};

template <typename T>
struct MpiReduction<Sum<T>> {
    static MPI_Op op() { return MPI_SUM; }
};

extern MPI_Op MPI_dual_min;
template<typename T>
struct MpiReduction<Min<Dual<T>>> {
    static MPI_Op op() { return MPI_dual_min; }
};

extern MPI_Op MPI_dual_max;
template<typename T>
struct MpiReduction<Max<Dual<T>>> {
    static MPI_Op op() { return MPI_dual_max; }
};

extern MPI_Op MPI_dual_sum;
template<typename T>
struct MpiReduction<Sum<Dual<T>>> {
    static MPI_Op op() { return MPI_dual_sum; }
};

// Allow ConservedQuantitiesNorm to be used as a custom scalar type for MPI reductions
extern MPI_Datatype MPI_ConservedQuantitiesNorm_real;
extern MPI_Datatype MPI_ConservedQuantitiesNorm_dual;

template <>
struct MpiDataType<ConservedQuantitiesNorm<Ibis::real>> {
    static MPI_Datatype value() { return MPI_ConservedQuantitiesNorm_real; }  
};
template <>
struct MpiDataType<ConservedQuantitiesNorm<Ibis::dual>> {
    static MPI_Datatype value() { return MPI_ConservedQuantitiesNorm_dual; }  
};

extern MPI_Op MPI_ConservedQuantitiesNorm_sum_real;
extern MPI_Op MPI_ConservedQuantitiesNorm_sum_dual;

template <>
struct MpiReduction<Sum<ConservedQuantitiesNorm<Ibis::real>>> {
    static MPI_Op op() { return MPI_ConservedQuantitiesNorm_sum_real; }
};
template <>
struct MpiReduction<Sum<ConservedQuantitiesNorm<Ibis::dual>>> {
    static MPI_Op op() { return MPI_ConservedQuantitiesNorm_sum_dual; }
};

void init_mpi_conserved_quantities_norms();
void init_mpi_dual();


// An object to perform MPI reductions with a nicer interfacer
template <class Reduction>
struct MpiReducer {
public:
    using Scalar = typename Reduction::scalar_type;
    using scalar_type = Scalar;
    using reduction = Reduction;

public:
    MpiReducer() : comm_(MPI_COMM_WORLD) {}

    MpiReducer(MPI_Comm comm) : comm_(comm) {}

    inline Scalar reduce(Scalar& local_value) {
        Scalar global_min;
        MPI_Datatype mpi_type = Ibis::MpiDataType<Scalar>::value();
        MPI_Allreduce(&local_value, &global_min, 1, mpi_type, mpi_op_, comm_);
        return global_min;
    }

    inline void reduce(Scalar* local_values, Scalar* global_values, size_t num_values) {
        MPI_Datatype mpi_type = Ibis::MpiDataType<Scalar>::value();
        MPI_Allreduce(local_values, global_values, num_values, mpi_type, mpi_op_, comm_);
    }

private:
    MPI_Comm comm_;
    MPI_Op mpi_op_ = MpiReduction<Reduction>::op();
};

template <class ReducerType>
class Reducer<ReducerType, Mpi> {
public:
    template <class FunctorType>
    inline auto execute(const std::string& str, const size_t work_count,
                        FunctorType functor) -> typename ReducerType::scalar_type {
        using scalar_type = typename ReducerType::scalar_type;

        // The shared memory part of the reduction
        scalar_type local_reduction =
            Ibis::parallel_reduce<ReducerType, SharedMem>(str, work_count, functor);

        // The distributed part of the reduction
        MpiReducer<ReducerType> mpi_reducer;
        return mpi_reducer.reduce(local_reduction);
    }

    template <class PolicyType, class FunctorType>
    inline auto execute(const std::string& str, PolicyType policy, FunctorType functor) ->
        typename ReducerType::scalar_type {
        using scalar_type = typename ReducerType::scalar_type;

        // The shared memory part of the reduction
        scalar_type local_reduction =
            Ibis::parallel_reduce<ReducerType, SharedMem>(str, policy, functor);

        // The distributed part of the reduction
        MpiReducer<ReducerType> mpi_reducer;
        return mpi_reducer.reduce(local_reduction);
    }
};

template <class ReducerType>
struct ReducerMap<ReducerType, Mpi> {
    using reducer_type = Reducer<ReducerType, Mpi>;
};

// Message passing communicator
template <typename T, bool gpu_aware = false, class MemSpace = Ibis::DefaultMemSpace>
class SymmetricComm {
private:
    using view_type = Kokkos::View<T*, MemSpace>;
    using mirror_view_type = typename view_type::host_mirror_type;

public:
    SymmetricComm(int other_rank, size_t buf_size)
        : other_rank_(other_rank), mpi_comm_(MPI_COMM_WORLD) {
        send_buf_ = view_type("send_buf", buf_size);
        recv_buf_ = view_type("recv_buf", buf_size);

        if constexpr (!gpu_aware) {
            host_send_buf_ = Kokkos::create_mirror_view(send_buf_);
            host_recv_buf_ = Kokkos::create_mirror_view(recv_buf_);
        }
    }

    SymmetricComm(int other_rank) : other_rank_(other_rank), mpi_comm_(MPI_COMM_WORLD) {}

    void expect_receive() {
        if (gpu_aware) {
            MPI_Irecv(recv_buf_.data(), recv_buf_.size(), mpi_type_, other_rank_, 0,
                      mpi_comm_, &recv_request_);
        } else {
            MPI_Irecv(host_recv_buf_.data(), host_recv_buf_.size(), mpi_type_,
                      other_rank_, 0, mpi_comm_, &recv_request_);
        }
    }

    void send() {
        if (gpu_aware) {
            MPI_Send(send_buf_.data(), send_buf_.size(), mpi_type_, other_rank_, 0,
                     mpi_comm_);
        } else {
            Kokkos::deep_copy(host_send_buf_, send_buf_);
            MPI_Send(host_send_buf_.data(), host_send_buf_.size(), mpi_type_, other_rank_,
                     0, mpi_comm_);
        }
    }

    MPI_Status receive() {
        MPI_Status recv_status;
        MPI_Wait(&recv_request_, &recv_status);
        if (!gpu_aware) {
            Kokkos::deep_copy(recv_buf_, host_recv_buf_);
        }
        return recv_status;
    }

    void resize_buffers(size_t new_size) {
        Kokkos::resize(send_buf_, new_size);
        Kokkos::resize(recv_buf_, new_size);
    }

    const Kokkos::View<T*, MemSpace>& send_buf() const { return send_buf_; }

    const Kokkos::View<T*, MemSpace>& recv_buf() const { return recv_buf_; }

private:
    // the send/receive buffers
    view_type send_buf_;
    view_type recv_buf_;
    mirror_view_type host_send_buf_;
    mirror_view_type host_recv_buf_;

    // some info for MPI
    MPI_Request recv_request_;
    MPI_Datatype mpi_type_ = Ibis::MpiDataType<T>::value();
    int other_rank_;
    MPI_Comm mpi_comm_;
};

}  // namespace Ibis

#endif
#endif
