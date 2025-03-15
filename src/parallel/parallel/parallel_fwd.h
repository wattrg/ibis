#ifndef IBIS_PARALLEL_FWD_H
#define IBIS_PARALLEL_FWD_H

#include <parallel/shared_memory_fwd.h>

#ifdef Ibis_ENABLE_MPI
#include "mpi/mpi/ibis_mpi_fwd.h"
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
