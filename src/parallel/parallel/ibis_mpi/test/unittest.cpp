#define DOCTEST_CONFIG_IMPLEMENT

#include <mpi.h>
#include <util/numeric_types.h>
#include <ibis_mpi/ibis_mpi.h>
#include <doctest/extensions/doctest_mpi.h>

#include <Kokkos_Core.hpp>

int main(int argc, char* argv[]) {
    // MPI_Init(&argc, &argv);
    doctest::mpi_init_thread(argc, argv, MPI_THREAD_MULTIPLE);
    Ibis::init_mpi_dual();
    Ibis::init_mpi_conserved_quantities_norms();

    Kokkos::initialize(argc, argv);


    doctest::Context ctx;
    ctx.setOption("reporters", "MpiConsoleReporter");

    ctx.applyCommandLine(argc, argv);
    int res = ctx.run();

    Kokkos::finalize();
    doctest::mpi_finalize();
    // MPI_Finalize();
    return res;
}
