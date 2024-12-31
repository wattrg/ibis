add_library(
    parallel
    STATIC
    parallel/distributed_memory.cpp
    parallel/shared_memory.cpp
)

target_link_libraries(
    parallel
    PUBLIC
    Kokkos::kokkos 
    doctest
    util
)

if (Ibis_ENABLE_MPI)
    add_subdirectory(parallel/mpi)
    target_link_libraries(
        parallel
        PUBLIC
        ibis_mpi
    )
endif(Ibis_ENABLE_MPI)

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
        parallel/shared_memory.cpp
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

    if (Ibis_ENABLE_MPI)
        target_link_libraries(
            parallel
            PUBLIC
            ibis_mpi
        )
        add_test(
            NAME parallel_unittest
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 parallel_unittest
        )
    else()
        add_test(
            NAME parallel_unittest
            COMMAND parallel_unittest
        )
    endif(Ibis_ENABLE_MPI)

    

endif(Ibis_BUILD_TESTS)
