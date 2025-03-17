add_subdirectory(parallel/ibis_kokkos)

add_library(
    parallel
    INTERFACE
    # parallel/shared_memory.cpp
)

target_link_libraries(
    parallel
    INTERFACE
    doctest
    util
    ibis_kokkos
)

target_include_directories(
    parallel
    INTERFACE
    .
)

if (Ibis_ENABLE_MPI)
    add_subdirectory(parallel/ibis_mpi)
    target_link_libraries(
        parallel
        INTERFACE
        ibis_mpi
    )
    target_include_directories(parallel INTERFACE .)
    target_compile_definitions(
        parallel
        INTERFACE
        Ibis_ENABLE_DISTRIBUTED_MEMORY
    )
endif(Ibis_ENABLE_MPI)


# if (Ibis_BUILD_TESTS)
#     add_executable(
#         parallel_unittest
#         test/unittest.cpp
#     )

#     target_include_directories(
#         parallel_unittest
#         PUBLIC
#         .
#     )

#     target_link_libraries(
#         parallel_unittest
#         PRIVATE
#         doctest
#         Kokkos::kokkos
#         parallel
#     )

#     add_test(NAME parallel_unittest COMMAND parallel_unittest)

# endif(Ibis_BUILD_TESTS)
