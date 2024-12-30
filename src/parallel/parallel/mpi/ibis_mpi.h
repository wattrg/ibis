#ifndef IBIS_MPI_H
#define IBIS_MPI_H

#ifdef Ibis_ENABLE_MPI

#include <mpi.h>
#include <parallel/reductions.h>

namespace Ibis{
namespace Distributed{

// Used as template parameter to determine which distributed memory
// paradigm to use
struct Mpi;

// MPI data types
template <typename Type>
struct MpiDataType;

template <typename Type>
struct MpiDataType;

#define MpiTypeMapping(type, MPI_type)                \
template <>                                           \
struct MpiDataType<type> {                            \
    static MPI_Datatype value() { return MPI_type; }  \
};                                                    \

MpiTypeMapping(short int, MPI_SHORT)
MpiTypeMapping(int, MPI_INT)
MpiTypeMapping(long int, MPI_LONG)
MpiTypeMapping(long long int, MPI_LONG_LONG)
MpiTypeMapping(unsigned char, MPI_UNSIGNED_CHAR)
MpiTypeMapping(unsigned short int, MPI_UNSIGNED_SHORT)
MpiTypeMapping(unsigned int, MPI_UNSIGNED)
MpiTypeMapping(unsigned long int, MPI_UNSIGNED_LONG)
MpiTypeMapping(unsigned long long int, MPI_UNSIGNED_LONG_LONG)
MpiTypeMapping(float, MPI_FLOAT)
MpiTypeMapping(double, MPI_DOUBLE)
MpiTypeMapping(long double, MPI_LONG_DOUBLE)
MpiTypeMapping(char, MPI_CHAR)


// Reductions
template <typename Reduction>
struct MpiReduction;

#define MpiReductionMapping(reduction, mpi_reduction)  \
template <>                                            \
struct MpiReduction<reduction> {                       \
    static MPI_Op op() { return mpi_reduction; }       \
};

MpiReductionMapping(Min, MPI_MIN)
MpiReductionMapping(Max, MPI_MAX)
MpiReductionMapping(Sum, MPI_SUM)


template <typename Scalar, class Reduction>
struct DistributedReduction {
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

public:
    using scalar_type = Scalar;
    using reduction = Reduction;

private:
    MPI_Comm comm_;
    MPI_Op mpi_op_ = MpiReduction<Reduction>::op();
};

template <typename Scalar>
using DistributedMin = DistributedReduction<Scalar, Min>;

template <typename Scalar>
using DistributedMax = DistributedReduction<Scalar, Max>;

template <typename Scalar>
using DistributedSum = DistributedReduction<Scalar, Sum>;

 
}
}

#endif
#endif
