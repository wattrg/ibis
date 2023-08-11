set(Kokkos_DIR "../../../kokkos_inst/lib/cmake/Kokkos")
set(Kokkos_ROOT "../../../kokkos_inst")
find_package(Kokkos REQUIRED)

add_library(grid STATIC src/grid.cpp src/grid_io.cpp src/vertex.cpp)
target_link_libraries(grid PUBLIC Kokkos::kokkos)
target_link_libraries(grid PRIVATE doctest)

add_executable(grid_unittest test/unittest.cpp src/grid_io.cpp)
target_link_libraries(grid_unittest PRIVATE doctest)
target_link_libraries(grid_unittest PRIVATE Kokkos::kokkos)
