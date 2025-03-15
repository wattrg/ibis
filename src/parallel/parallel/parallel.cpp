#include <parallel/parallel.h>
#include <parallel/shared_memory.h>
#include <parallel/distributed_memory.h>

namespace Ibis {

void initialise(int argc, char **argv) {

#ifdef Ibis_ENABLE_DISTRIBUTED_MEMORY
    Ibis::Distributed::initialise(argc, argv);  
#endif

    Ibis::Shared::initialise(argc, argv);
}

void finalise() {
    Ibis::Shared::finalise();

#ifdef Ibis_ENABLE_DISTRIBUTED_MEMORY
    Ibis::Distributed::finalise();
#endif
}
  
}
