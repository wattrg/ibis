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
	Kokkos::kokkos 
	doctest 
	util
	nlohmann_json::nlohmann_json
	spdlog::spdlog
	gas
)

if (METIS_FOUND)
    # target_include_directories(grid PRIVATE SYSTEM ${METIS_INCLUDE_PATH})
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
    )

    target_link_libraries(
        grid_unittest 
        PRIVATE 
        util 
        doctest 
        Kokkos::kokkos 
        nlohmann_json::nlohmann_json
        spdlog::spdlog
        grid
    )
   
    add_test(NAME grid_unittest COMMAND grid_unittest)
endif(Ibis_BUILD_TESTS)
