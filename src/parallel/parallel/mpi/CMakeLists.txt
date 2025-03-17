add_library(
    ibis_mpi
    STATIC
    mpi/ibis_mpi.cpp
)

target_link_libraries(
    ibis_mpi
    PUBLIC
    Kokkos::kokkos
    doctest
    util
    MPI::MPI_CXX
    parallel
)

target_include_directories(
    ibis_mpi
    PUBLIC
    .
)

target_compile_definitions(ibis_mpi PRIVATE Ibis_ENABLE_DISTRIBUTED_MEMORY)

if (Ibis_BUILD_TESTS)
    add_executable(
        mpi_unittest
        test/unittest.cpp
        mpi/ibis_mpi.cpp
    )
    target_include_directories(
        mpi_unittest
        PUBLIC
        .
    )

    target_link_libraries(
        mpi_unittest
        PRIVATE
        doctest
        Kokkos::kokkos
        parallel
        MPI::MPI_CXX
        parallel
    )
    add_test(
        NAME mpi_unittest
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 mpi_unittest
    )
endif(Ibis_BUILD_TESTS)
