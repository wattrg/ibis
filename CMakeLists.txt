cmake_minimum_required(VERSION 3.22.1)

project(ibis)

INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/cmake/build_env_info.cmake)
check_git_setup()


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INSTALL_PREFIX "../inst")
set (CMAKE_CXX_STANDARD 17)

add_subdirectory(src/gas)
add_subdirectory(src/util)
add_subdirectory(src/grid)
add_subdirectory(src/finite_volume)
add_subdirectory(src/ibis/src/commands)
add_subdirectory(src/solvers)

add_subdirectory(extern/doctest)
add_subdirectory(extern/kokkos)
add_subdirectory(extern/json)
add_subdirectory(extern/spdlog)

add_executable(ibis src/ibis/src/main.cpp)
target_include_directories(ibis PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(
	ibis 
	PUBLIC 
	prep 
	run 
	ibis_clean 
	gas 
	util 
	grid 
	doctest 
	ibis_git_version 
	spdlog::spdlog
)
target_compile_options(ibis PUBLIC -Werror -Wall -Wextra -Wpedantic)

install(TARGETS ${PROJECT_NAME} DESTINATION bin)
install(FILES src/ibis/src/commands/prep.py DESTINATION lib)
install(FILES src/ibis/src/commands/ibis_py_utils.py DESTINATION lib)
install(FILES src/ibis/src/commands/clean.py DESTINATION lib)
install(DIRECTORY resources DESTINATION .)

enable_testing()
add_test(NAME util_unittest COMMAND src/util/util_unittest)
add_test(NAME grid_unittest COMMAND src/grid/grid_unittest)
add_test(NAME finite_volume_unittest COMMAND src/finite_volume/finite_volume_unittest)
