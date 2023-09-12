add_library(solver STATIC src/solver.cpp src/solver.h)
# target_include_directories(solver PUBLIC ${CMAKE_SOURCE_DIR}/extern/json/single_include)
target_link_libraries(solver PUBLIC Kokkos::kokkos nlohmann_json::nlohmann_json)
# target_link_libraries(solver PUBLIC json)

add_library(runge_kutta STATIC src/runge_kutta.cpp src/solver.cpp)
target_link_libraries(runge_kutta PUBLIC Kokkos::kokkos doctest util nlohmann_json::nlohmann_json)
target_include_directories(runge_kutta PUBLIC ${CMAKE_SOURCE_DIR}/src)
