add_library(
	grid 
	STATIC 
	src/grid_io.cpp 
	src/vertex.cpp 
	src/interface.cpp 
	src/grid.cpp 
	src/cell.cpp
)

target_link_libraries(grid PUBLIC Kokkos::kokkos doctest util)

add_executable(
	grid_unittest 
	test/unittest.cpp 
	src/grid_io.cpp 
	src/grid.cpp 
	src/interface.cpp
	src/cell.cpp
	src/vertex.cpp
)
target_compile_options(grid_unittest PRIVATE -Wall -Wextra -Wpedantic -g)
target_link_libraries(grid_unittest PRIVATE util)
target_link_libraries(grid_unittest PRIVATE doctest)
target_link_libraries(grid_unittest PRIVATE Kokkos::kokkos)
