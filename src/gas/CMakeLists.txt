add_library(gas STATIC src/gas_state.cpp)
target_link_libraries(gas PUBLIC Kokkos::kokkos)
