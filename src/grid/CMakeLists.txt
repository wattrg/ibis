add_library(
	grid 
	STATIC 
	src/grid_io.cpp 
	src/vertex.cpp 
	src/interface.cpp 
	src/grid.cpp 
	src/cell.cpp
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

# target_include_directories(grid PUBLIC $CMAKE_SRC_DIRECTORY/src/)

if (Ibis_BUILD_TESTS)
    add_executable(
        grid_unittest 
        test/unittest.cpp 
        src/grid_io.cpp 
        src/grid.cpp 
        src/interface.cpp
        src/cell.cpp
        src/vertex.cpp
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
