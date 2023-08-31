cmake_minimum_required(VERSION 3.22.1)

project(ibis)

INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/cmake/build_env_info.cmake)
check_git_setup()


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INSTALL_PREFIX "../inst")

add_subdirectory(src/gas)
add_subdirectory(src/util)
add_subdirectory(src/grid)
add_subdirectory(src/finite_volume)
add_subdirectory(src/ibis/src/commands)
add_subdirectory(extern/doctest)
add_subdirectory(extern/kokkos)

add_executable(ibis src/ibis/src/main.cpp)
target_link_libraries(ibis PUBLIC prep)
target_compile_features(ibis PRIVATE cxx_std_20)
target_link_libraries(ibis PUBLIC gas)
target_link_libraries(ibis PUBLIC util)
target_link_libraries(ibis PUBLIC grid)
target_link_libraries(ibis PUBLIC Kokkos::kokkos)
target_link_libraries(ibis PRIVATE doctest)
target_link_libraries(ibis PRIVATE ibis_git_version)
target_compile_options(ibis PRIVATE -Wall -Wextra -Wpedantic)

install(TARGETS ${PROJECT_NAME} DESTINATION bin)
install(FILES src/ibis/src/commands/prep.py DESTINATION lib)

enable_testing()
add_test(NAME util_unittest COMMAND src/util/util_unittest)
add_test(NAME grid_unittest COMMAND src/grid/grid_unittest)
add_test(NAME finite_volume_unittest COMMAND src/finite_volume/finite_volume_unittest)
