#ifdef Ibis_ENABLE_MPI
#include <doctest/extensions/doctest_mpi.h>
#endif

#include <parallel/distributed_memory.h>
#include <util/numeric_types.h>

#ifdef Ibis_ENABLE_MPI

MPI_TEST_CASE("MPI_Min_scalar", 2) {
    double result = Ibis::Distributed::parallel_reduce<Min<double>>(
        "test", 10, KOKKOS_LAMBDA(const int i, double& utd) {
            utd = Ibis::min(utd, (double)test_rank + i);
        });

    MPI_CHECK(0, result == 0.0);
    MPI_CHECK(1, result == 0.0);
}

MPI_TEST_CASE("MPI_Sum_scalar", 2) {
    double result = Ibis::Distributed::parallel_reduce<Sum<double>>(
        "test", 10, KOKKOS_LAMBDA(const int i, double& utd) {
            utd += (double)i + (double)test_rank;
        });

    MPI_CHECK(0, result == 100.0);
    MPI_CHECK(1, result == 100.0);
}

MPI_TEST_CASE("MPI_Max_scalar", 2) {
    double result = Ibis::Distributed::parallel_reduce<Max<double>>(
        "test", 10, KOKKOS_LAMBDA(const int i, double& utd) {
            utd = Ibis::max(utd, (double)test_rank + i);
        });

    MPI_CHECK(0, result == 10.0);
    MPI_CHECK(0, result == 10.0);
}

#endif
