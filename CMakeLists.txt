cmake_minimum_required(VERSION 3.22.1)

project(ibis)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(src/gas)
add_subdirectory(src/util)
add_subdirectory(src/grid)
add_subdirectory(src/finite_volume)
add_subdirectory(extern/doctest)
add_subdirectory(extern/kokkos)

add_executable(${PROJECT_NAME} src/main.cpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
target_include_directories(${PROJECT_NAME} PRIVATE src/include)
target_link_libraries(${PROJECT_NAME} PUBLIC gas)
target_link_libraries(${PROJECT_NAME} PUBLIC util)
target_link_libraries(${PROJECT_NAME} PUBLIC grid)
target_link_libraries(${PROJECT_NAME} PUBLIC Kokkos::kokkos)
target_link_libraries(${PROJECT_NAME} PRIVATE doctest)
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)

enable_testing()
add_test(NAME util_unittest COMMAND src/util/util_unittest)
add_test(NAME grid_unittest COMMAND src/grid/grid_unittest)
