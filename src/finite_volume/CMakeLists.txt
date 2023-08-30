# add_library(finite_volume STATIC)
# target_link_libraries(grid PUBLIC Kokkos::kokkos)
# target_link_libraries(grid PRIVATE doctest)
# target_link_libraries(grid PRIVATE util)
#
add_executable(
	finite_volume_unittest 
	test/unittest.cpp 
	src/flux.cpp 
)
target_compile_options(finite_volume_unittest PRIVATE -Wall -Wextra -Wpedantic -Werror -g)
target_link_libraries(finite_volume_unittest PRIVATE util)
target_link_libraries(finite_volume_unittest PRIVATE doctest)
target_link_libraries(finite_volume_unittest PRIVATE Kokkos::kokkos)
