#define DOCTEST_CONFIG_IMPLEMENT

#include <mpi.h>
#include <doctest/doctest.h>

#include <Kokkos_Core.hpp>

int main(int argc, char* argv[]) {
    doctest::Context ctx;
    ctx.applyCommandLine(argc, argv);
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);
    int res = ctx.run();
    MPI_Finalize();
    Kokkos::finalize();
    return res;
}
