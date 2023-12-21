#include <doctest/doctest.h>
#include "Kokkos_Core_fwd.hpp"
#include "ragged_array.h"

using host_space = Kokkos::DefaultHostExecutionSpace::memory_space;
using host_layout = Kokkos::DefaultHostExecutionSpace::array_layout;

TEST_CASE("ragged_array") {
    std::vector<std::vector<double>> array = {
        {1.0, 2.0},
        {3.0, 4.0, 5.0},
        {6.0, 7.0, 8.0, 9.0}
    };    
    Ibis::RaggedArray<double> ragged_array ("array", array);
    auto ragged_array_host = ragged_array.host_mirror_and_copy();

    CHECK(ragged_array_host(0, 0) == doctest::Approx(1.0));
    CHECK(ragged_array_host(1, 2) == doctest::Approx(5.0));
    CHECK(ragged_array_host(0)(1) == doctest::Approx(2.0));
}
