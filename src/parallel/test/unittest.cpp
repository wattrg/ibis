#define DOCTEST_CONFIG_IMPLEMENT

#include <doctest/doctest.h>
#ifdef Ibis_ENABLE_MPI
#include <doctest/extensions/doctest_mpi.h>
#endif

#include <Kokkos_Core.hpp>

int main(int argc, char* argv[]) {
// MPI_Init(&argc, &argv);
#ifdef Ibis_ENABLE_MPI
    doctest::mpi_init_thread(argc, argv, MPI_THREAD_MULTIPLE);
#endif

    Kokkos::initialize(argc, argv);
    doctest::Context ctx;

    ctx.applyCommandLine(argc, argv);
    int res = ctx.run();

    Kokkos::finalize();

#ifdef Ibis_ENABLE_MPI
    doctest::mpi_finalize();
#endif

    return res;
}
