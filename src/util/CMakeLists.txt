set(Kokkos_DIR "../../../kokkos_inst/lib/cmake/Kokkos")
set(Kokkos_ROOT "../../../kokkos_inst")
find_package(Kokkos REQUIRED)
add_library(util STATIC vector3.cpp)
target_link_libraries(util PUBLIC Kokkos::kokkos)

add_executable(util_unittest unittest.cpp vector3.cpp)
target_link_libraries(util_unittest PRIVATE doctest::doctest)
target_link_libraries(util_unittest PUBLIC Kokkos::kokkos)
