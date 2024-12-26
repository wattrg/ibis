#include <doctest/doctest.h>
#include <parallel/distributed_memory.h>

TEST_CASE("MPI_Sum") {
    double result = Ibis::Distributed::parallel_reduce(
        "test", 10,
        KOKKOS_LAMBDA(const int i, double& utd){
            return 1.0;
    }, Ibis::Distributed::Min<double>());

    CHECK(result == 20.0);
}
