#define DOCTEST_CONFIG_IMPLEMENT

#include <mpi.h>
#include <util/numeric_types.h>
#include <ibis_mpi/ibis_mpi.h>
#include <doctest/extensions/doctest_mpi.h>

#include <Kokkos_Core.hpp>

int main(int argc, char* argv[]) {
    // MPI_Init(&argc, &argv);
    doctest::mpi_init_thread(argc, argv, MPI_THREAD_MULTIPLE);
    Kokkos::initialize(argc, argv);

    MPI_Op_create((MPI_User_function*)Ibis::MPI_custom_max<Ibis::dual>,
                  1, &Ibis::MPI_dual_max);
    MPI_Op_create((MPI_User_function*)Ibis::MPI_custom_min<Ibis::dual>,
                  1, &Ibis::MPI_dual_min);
    MPI_Op_create((MPI_User_function*)Ibis::MPI_custom_sum<Ibis::dual>,
                  1, &Ibis::MPI_dual_sum);
    Ibis::init_mpi_conserved_quantities_norm_sum<Ibis::real>(&Ibis::MPI_ConservedQuantitiesNorm,
                                           &Ibis::MPI_ConservedQuantitiesNorm_sum);
    Ibis::init_mpi_conserved_quantities_norm_sum<Ibis::dual>(&Ibis::MPI_ConservedQuantitiesNorm,
                                           &Ibis::MPI_ConservedQuantitiesNorm_sum);

    doctest::Context ctx;
    ctx.setOption("reporters", "MpiConsoleReporter");

    ctx.applyCommandLine(argc, argv);
    int res = ctx.run();

    Kokkos::finalize();
    doctest::mpi_finalize();
    // MPI_Finalize();
    return res;
}
