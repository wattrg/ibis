#include "parallel/ibis_mpi/ibis_mpi/ibis_mpi.h"

#include <doctest/extensions/doctest_mpi.h>
#include <ibis_mpi/ibis_mpi.h>
#include <mpi.h>
#include <parallel/parallel.h>
#include <util/numeric_types.h>

#include <Kokkos_Core.hpp>

#ifdef Ibis_ENABLE_MPI

namespace Ibis {
MPI_Op MPI_dual_min;
MPI_Op MPI_dual_max;
MPI_Op MPI_dual_sum;

MPI_Datatype MPI_ConservedQuantitiesNorm_real;
MPI_Datatype MPI_ConservedQuantitiesNorm_dual;
MPI_Op MPI_ConservedQuantitiesNorm_sum_real;
MPI_Op MPI_ConservedQuantitiesNorm_sum_dual;
};  // namespace Ibis

template <>
void Ibis::initialise<Mpi>(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    Ibis::init_mpi_dual();
    Ibis::init_mpi_conserved_quantities_norms();

    Ibis::initialise<SharedMem>(argc, argv);
}

template <>
void Ibis::finalise<Mpi>() {
    Ibis::finalise<SharedMem>();
    MPI_Finalize();
}

void Ibis::init_mpi_dual() {
    // Create MPI operations for dual numbers
    MPI_Op_create((MPI_User_function*)MPI_custom_max<Ibis::dual>, 1, &Ibis::MPI_dual_max);
    MPI_Op_create((MPI_User_function*)MPI_custom_min<Ibis::dual>, 1, &Ibis::MPI_dual_min);
    MPI_Op_create((MPI_User_function*)MPI_custom_sum<Ibis::dual>, 1, &Ibis::MPI_dual_sum);
}

void Ibis::init_mpi_conserved_quantities_norms() {
    MPI_Type_contiguous(6, MpiDataType<Ibis::real>::value(),
                        &MPI_ConservedQuantitiesNorm_real);
    MPI_Type_commit(&MPI_ConservedQuantitiesNorm_real);
    MPI_Op_create((MPI_User_function*)MPI_custom_sum<ConservedQuantitiesNorm<Ibis::real>>,
                  1, &MPI_ConservedQuantitiesNorm_sum_real);

    MPI_Type_contiguous(6, MpiDataType<Ibis::dual>::value(),
                        &MPI_ConservedQuantitiesNorm_dual);
    MPI_Type_commit(&MPI_ConservedQuantitiesNorm_dual);
    MPI_Op_create((MPI_User_function*)MPI_custom_sum<ConservedQuantitiesNorm<Ibis::dual>>,
                  1, &MPI_ConservedQuantitiesNorm_sum_dual);
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
    MPI_CHECK(1, result == 10.0);
}

// Message passing tests
MPI_TEST_CASE("MPI_comm", 2) {
    int other_rank = (test_rank == 0) ? 1 : 0;
    Ibis::SymmetricComm<Mpi, double, false> comm(other_rank, 10);

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

MPI_TEST_CASE("MPI_dual_min", 2) {
    Ibis::dual x;
    if (test_rank == 0) {
        x = Ibis::dual(1.0, 1.0);
    } else {
        x = Ibis::dual(2.0, 0.5);
    }

    Ibis::MpiReducer<Min<Ibis::dual>> mpi_min;
    Ibis::dual min = mpi_min.reduce(x);
    MPI_CHECK(0, min == Ibis::dual(1.0, 1.0));
}

MPI_TEST_CASE("MPI_dual_min_array", 2) {
    std::vector<Ibis::dual> xs;
    if (test_rank == 0) {
        xs = std::vector<Ibis::dual>{Ibis::dual(1.0, 1.0), Ibis::dual(0.5, 0.5)};
    } else {
        xs = std::vector<Ibis::dual>{Ibis::dual(2.0, 0.5), Ibis::dual(0.1, 1.0)};
    }

    Ibis::MpiReducer<Min<Ibis::dual>> mpi_min;
    std::vector<Ibis::dual> results(2);
    mpi_min.reduce(xs.data(), results.data(), 2);
    MPI_CHECK(0, results[0] == Ibis::dual(1.0, 1.0));
    MPI_CHECK(0, results[1] == Ibis::dual(0.1, 1.0));
}

MPI_TEST_CASE("MPI_dual_max", 2) {
    Ibis::dual x;

    if (test_rank == 0) {
        x = Ibis::dual(1.0, 1.0);
    } else {
        x = Ibis::dual(2.0, 0.5);
    }

    Ibis::MpiReducer<Max<Ibis::dual>> mpi_max;
    Ibis::dual max = mpi_max.reduce(x);
    MPI_CHECK(0, max == Ibis::dual(2.0, 0.5));
}

MPI_TEST_CASE("MPI_dual_max_lambda", 2) {
    Ibis::dual result = Ibis::parallel_reduce<Max<Ibis::dual>, Mpi>(
        "test", 10, KOKKOS_LAMBDA(const int i, Ibis::dual& utd) {
            Ibis::dual x{(double)test_rank + i, (double)test_rank + i + 1};
            utd = Ibis::max(utd, x);
        });

    MPI_CHECK(0, Ibis::real_part(result) == 10.0);
    MPI_CHECK(0, Ibis::dual_part(result) == 11.0);
    MPI_CHECK(1, Ibis::real_part(result) == 10.0);
    MPI_CHECK(1, Ibis::dual_part(result) == 11.0);
}

MPI_TEST_CASE("MPI_dual_sum", 2) {
    Ibis::dual x;

    if (test_rank == 0) {
        x = Ibis::dual(1.0, 1.0);
    } else {
        x = Ibis::dual(2.0, 0.5);
    }

    Ibis::MpiReducer<Sum<Ibis::dual>> mpi_sum;
    Ibis::dual sum = mpi_sum.reduce(x);
    MPI_CHECK(0, sum == Ibis::dual(3.0, 1.5));
    MPI_CHECK(1, sum == Ibis::dual(3.0, 1.5));
}

MPI_TEST_CASE("MPI_dual_sum_array", 2) {
    std::vector<Ibis::dual> xs;
    if (test_rank == 0) {
        xs = std::vector<Ibis::dual>{Ibis::dual(1.0, 1.0), Ibis::dual(0.5, 0.5)};
    } else {
        xs = std::vector<Ibis::dual>{Ibis::dual(2.0, 0.5), Ibis::dual(0.1, 1.0)};
    }

    Ibis::MpiReducer<Sum<Ibis::dual>> mpi_sum;
    std::vector<Ibis::dual> results(2);
    mpi_sum.reduce(xs.data(), results.data(), 2);
    MPI_CHECK(0, results[0] == Ibis::dual(3.0, 1.5));
    MPI_CHECK(0, results[1] == Ibis::dual(0.6, 1.5));
    MPI_CHECK(1, results[0] == Ibis::dual(3.0, 1.5));
    MPI_CHECK(1, results[1] == Ibis::dual(0.6, 1.5));
}

MPI_TEST_CASE("MPI_conserved_quantities_norm_sum_real", 2) {
    ConservedQuantitiesNorm<Ibis::real> x;
    if (test_rank == 0) {
        x.global() = 1.0;
        x.mass() = 0.1;
        x.momentum_x() = 0.2;
        x.momentum_y() = 0.3;
        x.momentum_z() = 0.4;
        x.energy() = 0.5;
    } else {
        x.global() = 2.0;
        x.mass() = 1.1;
        x.momentum_x() = 1.2;
        x.momentum_y() = 1.3;
        x.momentum_z() = 1.4;
        x.energy() = 1.5;
    }

    Ibis::MpiReducer<Sum<ConservedQuantitiesNorm<Ibis::real>>> mpi_sum;
    ConservedQuantitiesNorm<Ibis::real> sum = mpi_sum.reduce(x);
    CHECK(sum.global() == doctest::Approx(3.0));
    CHECK(sum.mass() == doctest::Approx(1.2));
    CHECK(sum.momentum_x() == doctest::Approx(1.4));
    CHECK(sum.momentum_y() == doctest::Approx(1.6));
    CHECK(sum.momentum_z() == doctest::Approx(1.8));
    CHECK(sum.energy() == doctest::Approx(2.0));
}

MPI_TEST_CASE("MPI_CQNorm_sum", 2) {
    ConservedQuantitiesNorm<Ibis::dual> x;
    if (test_rank == 0) {
        x.global() = Ibis::dual(1.0, 1.0);
        x.mass() = Ibis::dual(0.1, 0.1);
        x.momentum_x() = Ibis::dual(0.2, 0.2);
        x.momentum_y() = Ibis::dual(0.3, 0.3);
        x.momentum_z() = Ibis::dual(0.4, 0.4);
        x.energy() = Ibis::dual(0.5, 0.5);
    } else {
        x.global() = Ibis::dual(2.0, 2.0);
        x.mass() = Ibis::dual(1.1, 1.1);
        x.momentum_x() = Ibis::dual(1.2, 1.2);
        x.momentum_y() = Ibis::dual(1.3, 1.3);
        x.momentum_z() = Ibis::dual(1.4, 1.4);
        x.energy() = Ibis::dual(1.5, 1.5);
    }

    Ibis::MpiReducer<Sum<ConservedQuantitiesNorm<Ibis::dual>>> mpi_sum;
    ConservedQuantitiesNorm<Ibis::dual> sum = mpi_sum.reduce(x);
    CHECK(Ibis::real_part(sum.global()) == doctest::Approx(3.0));
    CHECK(Ibis::dual_part(sum.global()) == doctest::Approx(3.0));
    CHECK(Ibis::real_part(sum.mass()) == doctest::Approx(1.2));
    CHECK(Ibis::dual_part(sum.mass()) == doctest::Approx(1.2));
    CHECK(Ibis::real_part(sum.momentum_x()) == doctest::Approx(1.4));
    CHECK(Ibis::dual_part(sum.momentum_x()) == doctest::Approx(1.4));
    CHECK(Ibis::real_part(sum.momentum_y()) == doctest::Approx(1.6));
    CHECK(Ibis::dual_part(sum.momentum_y()) == doctest::Approx(1.6));
    CHECK(Ibis::real_part(sum.momentum_z()) == doctest::Approx(1.8));
    CHECK(Ibis::dual_part(sum.momentum_z()) == doctest::Approx(1.8));
    CHECK(Ibis::real_part(sum.energy()) == doctest::Approx(2.0));
    CHECK(Ibis::dual_part(sum.energy()) == doctest::Approx(2.0));
}

MPI_TEST_CASE("MPI_Sum_dual", 2) {
    Ibis::dual result = Ibis::parallel_reduce<Sum<Ibis::dual>, Mpi>(
        "test", 10, KOKKOS_LAMBDA(const int i, Ibis::dual& utd) {
            utd += Ibis::dual{(double)i + (double)test_rank, double(test_rank)};
        });

    CHECK(Ibis::real_part(result) == 100.0);
    CHECK(Ibis::dual_part(result) == 10.0);
}

MPI_TEST_CASE("MPI_sum_conserved_quantities_norm_dual", 2) {
    ConservedQuantitiesNorm<Ibis::dual> result =
        Ibis::parallel_reduce<Sum<ConservedQuantitiesNorm<Ibis::dual>>, Mpi>(
            "test", 10,
            KOKKOS_LAMBDA(const int i, ConservedQuantitiesNorm<Ibis::dual>& utd) {
                double id = (double)i;
                double rd = (double)test_rank;
                ConservedQuantitiesNorm<Ibis::dual> x;
                x.global() = Ibis::dual{id + rd, rd};
                x.mass() = Ibis::dual{id + rd + 1, rd + 1};
                x.momentum_x() = Ibis::dual{id + rd + 1, rd + 1};
                x.momentum_y() = Ibis::dual{id + rd + 2, rd + 2};
                x.momentum_z() = Ibis::dual{id + rd + 3, rd + 3};
                x.energy() = Ibis::dual{id + rd + 4, rd + 4};
                utd += x;
            });

    CHECK(Ibis::real_part(result.global()) == 100.0);
    CHECK(Ibis::dual_part(result.global()) == 10.0);
    CHECK(Ibis::real_part(result.mass()) == 120.0);
    CHECK(Ibis::dual_part(result.mass()) == 30.0);
    CHECK(Ibis::real_part(result.momentum_x()) == 120.0);
    CHECK(Ibis::dual_part(result.momentum_x()) == 30.0);
    CHECK(Ibis::real_part(result.momentum_y()) == 140.0);
    CHECK(Ibis::dual_part(result.momentum_y()) == 50.0);
    CHECK(Ibis::real_part(result.momentum_z()) == 160.0);
    CHECK(Ibis::dual_part(result.momentum_z()) == 70.0);
    CHECK(Ibis::real_part(result.energy()) == 180.0);
    CHECK(Ibis::dual_part(result.energy()) == 90.0);
}

MPI_TEST_CASE("MPI_sum_conserved_quantities_norm_real", 2) {
    ConservedQuantitiesNorm<Ibis::real> result =
        Ibis::parallel_reduce<Sum<ConservedQuantitiesNorm<Ibis::real>>, Mpi>(
            "test", 10,
            KOKKOS_LAMBDA(const int i, ConservedQuantitiesNorm<Ibis::real>& utd) {
                double id = (double)i;
                double rd = (double)test_rank;
                ConservedQuantitiesNorm<Ibis::real> x;
                x.global() = id + rd;
                x.mass() = id + rd + 1;
                x.momentum_x() = id + rd + 1;
                x.momentum_y() = id + rd + 2;
                x.momentum_z() = id + rd + 3;
                x.energy() = id + rd + 4;
                utd += x;
            });

    CHECK(result.global() == 100.0);
    CHECK(result.mass() == 120.0);
    CHECK(result.momentum_x() == 120.0);
    CHECK(result.momentum_y() == 140.0);
    CHECK(result.momentum_z() == 160.0);
    CHECK(result.energy() == 180.0);
}

#endif  // DOCTEST_CONFIG_DISABLE

#endif
