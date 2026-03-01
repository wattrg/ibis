add_library(
  graph_colouring_kokkos_kernels
  STATIC
  graph_colouring_kokkos_kernels/graph_colouring_kokkos_kernels.cpp
)

target_include_directories(
  graph_colouring_kokkos_kernels
  PUBLIC
  .
)

target_link_libraries(
  graph_colouring_kokkos_kernels
  PUBLIC
  grid
  Kokkos::kokkos
  KokkosKernels::kokkoskernels
  doctest
  graph_colouring
)

if (Ibis_BUILD_TESTS)
  add_executable(
    graph_colouring_kokkos_kernels_unittest
    test/unittest.cpp
    graph_colouring_kokkos_kernels/graph_colouring_kokkos_kernels.cpp
  )

  target_link_libraries(
    graph_colouring_kokkos_kernels_unittest
    PRIVATE
    util
    graph_colouring_kokkos_kernels
  )

  add_test(NAME graph_colouring_kokkos_kernels COMMAND graph_colouring_kokkos_kernels_unittest)
endif (Ibis_BUILD_TESTS)
