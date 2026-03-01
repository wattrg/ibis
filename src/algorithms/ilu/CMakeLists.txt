add_subdirectory(ilu_kokkos_kernels)

add_library(ilu INTERFACE)
target_include_directories(ilu INTERFACE .)
target_link_libraries(ilu INTERFACE ilu_kokkos_kernels)
