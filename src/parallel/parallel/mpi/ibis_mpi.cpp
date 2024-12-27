#include <doctest/extensions/doctest_mpi.h>
#include <Kokkos_Core.hpp>
#include <parallel/mpi/ibis_mpi.h>
#include <util/numeric_types.h>
#include <vector>

#ifdef Ibis_ENABLE_MPI

MPI_TEST_CASE("MPI_Min_scalar", 2) {
    // double result = Ibis::Distributed::parallel_reduce(
    //     "test", 10,
    //     KOKKOS_LAMBDA(const int i, double& utd){
    //         (void)i;
    //         (void)utd;
    //         return 1.0;
    // }, Ibis::Distributed::Min<double>());
    Ibis::Distributed::DistributedMin<double> mpi_min;

    double x = 1.0 + test_rank;
    MPI_CHECK(0, x == 1.0);
    MPI_CHECK(1, x == 2.0);

    double global_min = mpi_min.reduce(x);
    MPI_CHECK(0, global_min == 1.0);
    MPI_CHECK(1, global_min == 1.0);
}

MPI_TEST_CASE("MPI_Min_array", 2) {
    Ibis::Distributed::DistributedMin<double> mpi_min;

    std::vector<double> local_values {1.0 + test_rank, 2.0 + test_rank};
    std::vector<double> global_values (2);

    mpi_min.reduce(local_values.data(), global_values.data(), 2);
    MPI_CHECK(0, global_values == std::vector<double> { 1.0, 2.0});
    MPI_CHECK(1, global_values == std::vector<double> { 1.0, 2.0});
    
}

MPI_TEST_CASE("MPI_Sum_scalar", 2) {
    // double result = Ibis::Distributed::parallel_reduce(
    //     "test", 10,
    //     KOKKOS_LAMBDA(const int i, double& utd){
    //         (void)i;
    //         (void)utd;
    //         return 1.0;
    // }, Ibis::Distributed::Min<double>());
    Ibis::Distributed::DistributedSum<double> mpi_sum;

    double x = 1.0 + test_rank;
    MPI_CHECK(0, x == 1.0);
    MPI_CHECK(1, x == 2.0);

    double global_sum = mpi_sum.reduce(x);
    MPI_CHECK(0, global_sum == 3.0);
    MPI_CHECK(1, global_sum == 3.0);
}

MPI_TEST_CASE("MPI_Min_array", 2) {
    Ibis::Distributed::DistributedSum<double> mpi_sum;

    std::vector<double> local_values {1.0 + test_rank, 2.0 + test_rank};
    std::vector<double> global_values (2);

    mpi_sum.reduce(local_values.data(), global_values.data(), 2);
    MPI_CHECK(0, global_values == std::vector<double> { 3.0, 5.0});
    MPI_CHECK(1, global_values == std::vector<double> { 3.0, 5.0});
    
}

#endif
