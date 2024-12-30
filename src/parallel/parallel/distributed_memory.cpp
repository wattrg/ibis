#include <doctest/extensions/doctest_mpi.h>
#include <parallel/distributed_memory.h>
#include <util/numeric_types.h>

MPI_TEST_CASE("MPI_Min_scalar", 2) {
    double result = Ibis::Distributed::parallel_reduce(
        "test", 10,
        KOKKOS_LAMBDA(const int i, double& utd){
            (void)i;
            (void)utd;
            utd = Ibis::min(utd, (double)test_rank);
    }, Ibis::Distributed::DistributedMin<double>());

    MPI_CHECK(0, result == 0.0);
    MPI_CHECK(1, result == 0.0);
}
