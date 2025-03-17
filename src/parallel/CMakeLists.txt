add_library(
    parallel
    STATIC
    parallel/shared_memory.cpp
)

target_link_libraries(
    parallel
    PUBLIC
    Kokkos::kokkos 
    doctest
    util
)

target_include_directories(
    parallel
    PUBLIC
    .
)

if (Ibis_ENABLE_MPI)
    add_subdirectory(parallel/mpi)
    target_link_libraries(
        parallel
        PRIVATE
        ibis_mpi
    )
    target_include_directories(parallel PUBLIC .)
    target_compile_definitions(
        parallel
        PRIVATE
        Ibis_ENABLE_DISTRIBUTED_MEMORY
    )
endif(Ibis_ENABLE_MPI)


if (Ibis_BUILD_TESTS)
    add_executable(
        parallel_unittest
        test/unittest.cpp
    )

    target_include_directories(
        parallel_unittest
        PUBLIC
        .
    )

    target_link_libraries(
        parallel_unittest
        PRIVATE
        doctest
        Kokkos::kokkos
        parallel
    )

    add_test(NAME parallel_unittest COMMAND parallel_unittest)

endif(Ibis_BUILD_TESTS)
