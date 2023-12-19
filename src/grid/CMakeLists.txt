add_library(
	grid 
	STATIC 
	grid/grid_io.cpp 
	grid/vertex.cpp 
	grid/interface.cpp 
	grid/grid.cpp 
	grid/cell.cpp
)

target_link_libraries(
	grid 
	PUBLIC 
	Kokkos::kokkos 
	doctest 
	util
	nlohmann_json::nlohmann_json
	spdlog::spdlog
)

target_include_directories(
    grid
    PUBLIC
    .
)

# target_include_directories(grid PUBLIC $CMAKE_SRC_DIRECTORY/src/)

if (Ibis_BUILD_TESTS)
    add_executable(
        grid_unittest 
        test/unittest.cpp 
        grid/grid_io.cpp 
        grid/grid.cpp 
        grid/interface.cpp
        grid/cell.cpp
        grid/vertex.cpp
    )
    target_link_libraries(
        grid_unittest 
        PRIVATE 
        util 
        doctest 
        Kokkos::kokkos 
        nlohmann_json::nlohmann_json
        spdlog::spdlog
    )
    
    add_test(NAME grid_unittest COMMAND grid_unittest)
endif(Ibis_BUILD_TESTS)
