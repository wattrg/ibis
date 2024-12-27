#define DOCTEST_CONFIG_IMPLEMENT

#include <mpi.h>
// #include <doctest/doctest.h>
#include <doctest/extensions/doctest_mpi.h>

#include <Kokkos_Core.hpp>

int main(int argc, char* argv[]) {
    // MPI_Init(&argc, &argv);
    doctest::mpi_init_thread(argc,argv,MPI_THREAD_MULTIPLE);
    doctest::Context ctx;
    Kokkos::initialize(argc, argv);
    ctx.applyCommandLine(argc, argv);
    int res = ctx.run();
    // MPI_Finalize();
    doctest::mpi_finalize();
    Kokkos::finalize();
    return res;
}
