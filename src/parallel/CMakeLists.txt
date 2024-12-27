add_library(
    parallel
    STATIC
    parallel/distributed_memory.cpp
    parallel/mpi/ibis_mpi.cpp
)

target_link_libraries(
    parallel
    PUBLIC
    Kokkos::kokkos 
    doctest
    util
    MPI::MPI_CXX
)

target_include_directories(
    parallel
    PUBLIC
    .
)

if (Ibis_BUILD_TESTS)
    add_executable(
        parallel_unittest
        test/unittest.cpp
        parallel/distributed_memory.cpp
        parallel/mpi/ibis_mpi.cpp
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
        MPI::MPI_CXX
    )

    add_test(NAME parallel_unittest COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 parallel_unittest)
endif(Ibis_BUILD_TESTS)
