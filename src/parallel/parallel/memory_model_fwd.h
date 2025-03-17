#ifndef MEMORY_MODEL_FWD_H
#define MEMORY_MODEL_FWD_H

#include <ibis_kokkos/ibis_kokkos_fwd.h>

#ifdef Ibis_ENABLE_MPI
#include <ibis_mpi/ibis_mpi_fwd.h>
#endif

namespace Ibis {

#ifdef Ibis_ENABLE_DISTRIBUTED_MEMORY
#ifdef Ibis_ENABLE_MPI
using DefaultMemModel = Mpi;
#endif
#else
using DefaultMemModel = SharedMem;
#endif

}  // namespace Ibis

#endif
