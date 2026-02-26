#ifndef IBIS_MPI_CONSERVED_QUANTITIES_H
#define IBIS_MPI_CONSERVED_QUANTITIES_H

#include <ibis_mpi/ibis_mpi_types.h>
#include <mpi.h>
#include <util/conserved_quantities.h>

namespace Ibis {
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
void finalise_mpi_conserved_quantities_norms();
}  // namespace Ibis

#endif
