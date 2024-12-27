#include <doctest/extensions/doctest_mpi.h>
#include <Kokkos_Core.hpp>
#include <parallel/mpi/ibis_mpi.h>
#include <util/numeric_types.h>
#include <vector>

#ifdef Ibis_ENABLE_MPI

template <typename Scalar>
Scalar Ibis::Distributed::Min<Scalar>::reduce(Scalar& local_value) {
    Scalar global_min;
    MPI_Datatype mpi_type = Ibis::Distributed::MpiDataType<Scalar>::value();
    MPI_Allreduce(&local_value, &global_min, 1, mpi_type, MPI_MIN, comm_);
    return global_min;
}

template <typename Scalar>
void Ibis::Distributed::Min<Scalar>::reduce(Scalar* local_values,
                                            Scalar* global_values,
                                            size_t num_values) {
    MPI_Datatype mpi_type = Ibis::Distributed::MpiDataType<Scalar>::value();
    MPI_Allreduce(local_values, global_values, num_values, mpi_type, MPI_MIN, comm_);
}
template struct Ibis::Distributed::Min<Ibis::real>;


MPI_TEST_CASE("MPI_Min_scalar", 2) {
    // double result = Ibis::Distributed::parallel_reduce(
    //     "test", 10,
    //     KOKKOS_LAMBDA(const int i, double& utd){
    //         (void)i;
    //         (void)utd;
    //         return 1.0;
    // }, Ibis::Distributed::Min<double>());
    Ibis::Distributed::Min<double> mpi_min;

    double x = 1.0 + test_rank;
    MPI_CHECK(0, x == 1.0);
    MPI_CHECK(1, x == 2.0);

    double global_min = mpi_min.reduce(x);
    MPI_CHECK(0, global_min == 1.0);
    MPI_CHECK(1, global_min == 1.0);
}

MPI_TEST_CASE("MPI_Min_array", 2) {
    Ibis::Distributed::Min<double> mpi_min;

    std::vector<double> local_values {1.0 + test_rank, 2.0 + test_rank};
    std::vector<double> global_values (2);

    mpi_min.reduce(local_values.data(), global_values.data(), 2);
    MPI_CHECK(0, global_values == std::vector<double> { 1.0, 2.0});
    MPI_CHECK(1, global_values == std::vector<double> { 1.0, 2.0});
    
}

#endif
