#include <doctest/doctest.h>
// #include <parallel/shared_memory.h>
#include <util/numeric_types.h>
#include <Kokkos_Core.hpp>
#include <parallel/parallel.h>

template <>
void Ibis::initialise<SharedMem>(int argc, char **argv) {
    Kokkos::initialize(argc, argv);
}

template <>
void Ibis::finalise<SharedMem>() {
    Kokkos::finalize();
}

TEST_CASE("shared_parallel_reduction") {
    double result = Ibis::parallel_reduce<Min<double>, SharedMem>(
        "test", 10,
        KOKKOS_LAMBDA(const int i, double &utd) { utd = Ibis::min(utd, (double)i); });

    CHECK(result == 0.0);
}

TEST_CASE("shared_parallel_reduction") {
    double result = Ibis::parallel_reduce<Min<double>, SharedMem>(
        "test", Kokkos::RangePolicy(5, 10),
        KOKKOS_LAMBDA(const int i, double &utd) { utd = Ibis::min(utd, (double)i); });

    CHECK(result == 5);
}
