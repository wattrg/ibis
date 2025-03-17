add_library(
    ibis_kokkos
    STATIC
    ibis_kokkos/ibis_kokkos.cpp
)

target_link_libraries(
    ibis_kokkos
    PUBLIC
    Kokkos::kokkos
    doctest
    util
    parallel
)

target_include_directories(
    ibis_kokkos
    PUBLIC
    .
)

if (Ibis_BUILD_TESTS)
    add_executable(
        kokkos_unittest
        test/unittest.cpp
    )
    target_include_directories(
        kokkos_unittest
        PUBLIC
        .
    )

    target_link_libraries(
        kokkos_unittest
        PRIVATE
        ibis_kokkos
    )
    add_test(
        NAME kokkos_unittest
        COMMAND kokkos_unittest
    )
endif(Ibis_BUILD_TESTS)
