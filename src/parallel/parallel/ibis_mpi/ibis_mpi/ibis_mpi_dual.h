#ifndef IBIS_MPI_DUAL_H
#define IBIS_MPI_DUAL_H

#include <ibis_mpi/ibis_mpi_types.h>
#include <mpi.h>
#include <util/numeric_types.h>

namespace Ibis {
template <>
struct MpiDataType<Dual<double>> {
    static MPI_Datatype value() { return MPI_2DOUBLE_PRECISION; }
};

template <>
struct MpiDataType<Dual<float>> {
    static MPI_Datatype value() { return MPI_2REAL; }
};

extern MPI_Op MPI_dual_min;
template <typename T>
struct MpiReduction<Min<Dual<T>>> {
    static MPI_Op op() { return MPI_dual_min; }
};

extern MPI_Op MPI_dual_max;
template <typename T>
struct MpiReduction<Max<Dual<T>>> {
    static MPI_Op op() { return MPI_dual_max; }
};

extern MPI_Op MPI_dual_sum;
template <typename T>
struct MpiReduction<Sum<Dual<T>>> {
    static MPI_Op op() { return MPI_dual_sum; }
};

void init_mpi_dual();
void finalise_mpi_dual();
}  // namespace Ibis

#endif
