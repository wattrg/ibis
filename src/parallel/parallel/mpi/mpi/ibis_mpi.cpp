#include <doctest/extensions/doctest_mpi.h>
#include <mpi.h>
#include <mpi/ibis_mpi.h>
#include <parallel/parallel.h>
#include <util/numeric_types.h>

#include <Kokkos_Core.hpp>

#ifdef Ibis_ENABLE_MPI

template <>
void Ibis::initialise<Mpi>(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    Ibis::initialise<SharedMem>(argc, argv);
}

template <>
void Ibis::finalise<Mpi>() {
    MPI_Finalize();
}

#ifndef DOCTEST_CONFIG_DISABLE
// Tests
// Pure MPI reductions
MPI_TEST_CASE("MPI_Min_scalar", 2) {
    Ibis::MpiReducer<Min<double>> mpi_min;

    double x = 1.0 + test_rank;
    MPI_CHECK(0, x == 1.0);
    MPI_CHECK(1, x == 2.0);

    double global_min = mpi_min.reduce(x);
    MPI_CHECK(0, global_min == 1.0);
    MPI_CHECK(1, global_min == 1.0);
}

MPI_TEST_CASE("MPI_Min_array", 2) {
    Ibis::MpiReducer<Min<double>> mpi_min;

    std::vector<double> local_values{1.0 + test_rank, 2.0 + test_rank};
    std::vector<double> global_values(2);

    mpi_min.reduce(local_values.data(), global_values.data(), 2);
    MPI_CHECK(0, global_values == std::vector<double>{1.0, 2.0});
    MPI_CHECK(1, global_values == std::vector<double>{1.0, 2.0});
}

MPI_TEST_CASE("MPI_Sum_scalar", 2) {
    Ibis::MpiReducer<Sum<double>> mpi_sum;

    double x = 1.0 + test_rank;
    MPI_CHECK(0, x == 1.0);
    MPI_CHECK(1, x == 2.0);

    double global_sum = mpi_sum.reduce(x);
    MPI_CHECK(0, global_sum == 3.0);
    MPI_CHECK(1, global_sum == 3.0);
}

MPI_TEST_CASE("MPI_Sum_array", 2) {
    Ibis::MpiReducer<Sum<double>> mpi_sum;

    std::vector<double> local_values{1.0 + test_rank, 2.0 + test_rank};
    std::vector<double> global_values(2);

    mpi_sum.reduce(local_values.data(), global_values.data(), 2);
    MPI_CHECK(0, global_values == std::vector<double>{3.0, 5.0});
    MPI_CHECK(1, global_values == std::vector<double>{3.0, 5.0});
}

// Mixed shared and MPI reductions
MPI_TEST_CASE("MPI_Min_scalar", 2) {
    double result = Ibis::parallel_reduce<Min<double>, Mpi>(
        "test", 10, KOKKOS_LAMBDA(const int i, double& utd) {
            utd = Ibis::min(utd, (double)test_rank + i);
        });

    MPI_CHECK(0, result == 0.0);
    MPI_CHECK(1, result == 0.0);
}

MPI_TEST_CASE("MPI_Sum_scalar", 2) {
    double result = Ibis::parallel_reduce<Sum<double>, Mpi>(
        "test", 10, KOKKOS_LAMBDA(const int i, double& utd) {
            utd += (double)i + (double)test_rank;
        });

    MPI_CHECK(0, result == 100.0);
    MPI_CHECK(1, result == 100.0);
}

MPI_TEST_CASE("MPI_Max_scalar", 2) {
    double result = Ibis::parallel_reduce<Max<double>, Mpi>(
        "test", 10, KOKKOS_LAMBDA(const int i, double& utd) {
            utd = Ibis::max(utd, (double)test_rank + i);
        });

    MPI_CHECK(0, result == 10.0);
    MPI_CHECK(0, result == 10.0);
}

// Message passing tests
MPI_TEST_CASE("MPI_comm", 2) {
    int other_rank = (test_rank == 0) ? 1 : 0;
    Ibis::SymmetricComm<double> comm(other_rank, 10);

    comm.expect_receive();

    auto send_buf = comm.send_buf();
    auto send_buf_mirror = Kokkos::create_mirror_view(send_buf);
    for (int i = 0; i < 10; i++) {
        send_buf_mirror(i) = (double)test_rank + (double)i;
    }
    Kokkos::deep_copy(send_buf, send_buf_mirror);

    comm.send();
    comm.receive();

    auto recv_buf = comm.recv_buf();
    auto recv_buf_mirror = Kokkos::create_mirror_view(recv_buf);
    Kokkos::deep_copy(recv_buf_mirror, recv_buf);
    for (int i = 0; i < 10; i++) {
        MPI_CHECK(0, recv_buf_mirror(i) == 1.0 + (double)i);
        MPI_CHECK(1, recv_buf_mirror(i) == 0.0 + (double)i);
    }
}
#endif  // DOCTEST_CONFIG_DISABLE

#endif
