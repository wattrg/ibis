#define DOCTEST_CONFIG_IMPLEMENT

#include <mpi.h>
// #include <doctest/doctest.h>
#include <doctest/extensions/doctest_mpi.h>

#include <Kokkos_Core.hpp>

int main(int argc, char* argv[]) {
    // MPI_Init(&argc, &argv);
    doctest::mpi_init_thread(argc, argv, MPI_THREAD_MULTIPLE);
    Kokkos::initialize(argc, argv);
    doctest::Context ctx;

    ctx.applyCommandLine(argc, argv);
    int res = ctx.run();

    Kokkos::finalize();
    doctest::mpi_finalize();
    // MPI_Finalize();
    return res;
}
