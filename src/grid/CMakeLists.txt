add_library(
	grid 
	STATIC 
	grid/grid_io.cpp 
	grid/vertex.cpp 
	grid/interface.cpp 
	grid/grid.cpp 
	grid/cell.cpp
	grid/geom.cpp
	grid/gradient.cpp
	grid/partition.cpp
)

target_include_directories(
    grid
    PUBLIC
    .
)

target_link_libraries(
	grid 
	PUBLIC 
  doctest 
	util
	nlohmann_json::nlohmann_json
	spdlog::spdlog
	gas
	parallel
)

if (METIS_FOUND)
    target_link_libraries(grid PRIVATE ${METIS_LIBRARIES})
endif(METIS_FOUND)

if (Ibis_BUILD_TESTS)
    add_executable(
        grid_unittest 
        test/unittest.cpp 
        grid/grid_io.cpp 
        grid/grid.cpp 
        grid/interface.cpp
        grid/cell.cpp
        grid/vertex.cpp
        grid/geom.cpp
        grid/gradient.cpp
        grid/partition.cpp
    )

    target_link_libraries(
        grid_unittest 
        PRIVATE 
        util 
        doctest 
        nlohmann_json::nlohmann_json
        spdlog::spdlog
        grid
    )

    if (Ibis_ENABLE_MPI)
      add_executable(
        grid_mpi_unittest
        test/mpi_unittest.cpp
        test/grid_mpi_unittest.cpp
      )

      target_link_libraries(
        grid_mpi_unittest
        PRIVATE
        grid
        doctest
      )

      add_test(
        NAME grid_mpi_unittest
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 grid_mpi_unittest
      )
    endif(Ibis_ENABLE_MPI)

    add_test(NAME grid_unittest COMMAND grid_unittest)
endif(Ibis_BUILD_TESTS)
