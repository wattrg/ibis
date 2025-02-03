#include <doctest/doctest.h>
#include <parallel/shared_memory.h>
#include <util/numeric_types.h>
#include "Kokkos_Core.hpp"

void Ibis::Shared::initialise(int argc, char **argv) {
    Kokkos::initialize(argc, argv);
}

void Ibis::Shared::finalise() {
    Kokkos::finalize();
}

TEST_CASE("shared_parallel_reduction") {
    double result = Ibis::Shared::parallel_reduce<Min<double>>(
        "test", 10,
        KOKKOS_LAMBDA(const int i, double &utd) { utd = Ibis::min(utd, (double)i); });

    CHECK(result == 0.0);
}
