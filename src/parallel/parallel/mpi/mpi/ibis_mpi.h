#ifndef IBIS_MPI_H
#define IBIS_MPI_H

#ifdef Ibis_ENABLE_MPI

#include <mpi.h>
#include <parallel/reductions.h>

#include <Kokkos_Core.hpp>

namespace Ibis {
namespace Distributed {

// Used as template parameter to determine which distributed memory
// paradigm to use
struct Mpi;

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

    // Reductions
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

template <class Reduction>
struct DistributedReduction {
public:
    using Scalar = typename Reduction::scalar_type;
    using scalar_type = Scalar;
    using reduction = Reduction;

public:
    DistributedReduction() : comm_(MPI_COMM_WORLD) {}

    DistributedReduction(MPI_Comm comm) : comm_(comm) {}

    Scalar reduce(Scalar& local_value) {
        Scalar global_min;
        MPI_Datatype mpi_type = Ibis::Distributed::MpiDataType<Scalar>::value();
        MPI_Allreduce(&local_value, &global_min, 1, mpi_type, mpi_op_, comm_);
        return global_min;
    }

    void reduce(Scalar* local_values, Scalar* global_values, size_t num_values) {
        MPI_Datatype mpi_type = Ibis::Distributed::MpiDataType<Scalar>::value();
        MPI_Allreduce(local_values, global_values, num_values, mpi_type, mpi_op_, comm_);
    }

private:
    MPI_Comm comm_;
    MPI_Op mpi_op_ = MpiReduction<Reduction>::op();
};

template <typename Scalar>
using DistributedMin = DistributedReduction<Min<Scalar>>;

template <typename Scalar>
using DistributedMax = DistributedReduction<Max<Scalar>>;

template <typename Scalar>
using DistributedSum = DistributedReduction<Sum<Scalar>>;

template <typename T, class MemSpace = Kokkos::DefaultExecutionSpace::memory_space>
class SymmetricComm {
public:
    SymmetricComm(int other_rank, size_t buf_size)
        : other_rank_(other_rank), mpi_comm_(MPI_COMM_WORLD) {
        send_buf_ = Kokkos::View<T*, MemSpace>("send_buf", buf_size);
        recv_buf_ = Kokkos::View<T*, MemSpace>("recv_buf", buf_size);
    }

    SymmetricComm(int other_rank) : other_rank_(other_rank), mpi_comm_(MPI_COMM_WORLD) {}

    void expect_receive() {
        MPI_Irecv(recv_buf_.data(), recv_buf_.size(), mpi_type_, other_rank_, 0,
                  mpi_comm_, &recv_request_);
    }

    void send() {
        MPI_Send(send_buf_.data(), send_buf_.size(), mpi_type_, other_rank_, 0,
                 mpi_comm_);
    }

    MPI_Status receive() {
        MPI_Status recv_status;
        MPI_Wait(&recv_request_, &recv_status);
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
    Kokkos::View<T*, MemSpace> send_buf_;
    Kokkos::View<T*, MemSpace> recv_buf_;

    // some info for MPI
    MPI_Request recv_request_;
    MPI_Datatype mpi_type_ = Ibis::Distributed::MpiDataType<T>::value();
    int other_rank_;
    MPI_Comm mpi_comm_;
};

}  // namespace Distributed
}  // namespace Ibis

#endif
#endif
