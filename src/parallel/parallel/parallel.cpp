#include <parallel/parallel.h>
#include <parallel/shared_memory.h>
#include <parallel/distributed_memory.h>

namespace Ibis {

// template <class MemModel>
// void initialise(int argc, char **argv) {

// #ifdef Ibis_ENABLE_DISTRIBUTED_MEMORY
//     Ibis::Distributed::initialise(argc, argv);  
// #endif

//     Ibis::Shared::initialise(argc, argv);
// }

// template <class MemModel>
// void finalise() {
//     Ibis::Shared::finalise();

// #ifdef Ibis_ENABLE_MPI
//     Ibis::Distributed::finalise();
// #endif
// }
  
}
