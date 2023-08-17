add_library(util STATIC src/vector3.cpp src/id.cpp)
target_link_libraries(util PUBLIC Kokkos::kokkos)
target_link_libraries(util PRIVATE doctest)
target_compile_options(util PUBLIC -Wall -Wextra -Wpedantic -g)

add_executable(util_unittest test/unittest.cpp src/vector3.cpp src/id.cpp)
target_compile_options(util_unittest PRIVATE -Wall -Wextra -Wpedantic -g)
target_link_libraries(util_unittest PRIVATE doctest)
target_link_libraries(util_unittest PUBLIC Kokkos::kokkos)
