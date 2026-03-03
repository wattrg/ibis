add_library(
  triangular_solver_kokkos_kernels
  STATIC
  triangular_solver_kokkos_kernels/triangular_solver_kokkos_kernels.cpp
)

target_include_directories(
  triangular_solver_kokkos_kernels
  PUBLIC
  .
)

target_link_libraries(
  triangular_solver_kokkos_kernels
  PUBLIC
  linear_algebra
  Kokkos::kokkos
  KokkosKernels::kokkoskernels
  doctest
  triangular_solver
)

if (Ibis_BUILD_TESTS)
  add_executable(
    triangular_solver_kokkos_kernels_unittest
    test/unittest.cpp
    triangular_solver_kokkos_kernels/triangular_solver_kokkos_kernels.cpp
  )

  target_link_libraries(
    triangular_solver_kokkos_kernels_unittest
    PRIVATE
    util
    triangular_solver_kokkos_kernels
  )

  add_test(NAME triangular_solver_kokkos_kernels COMMAND triangular_solver_kokkos_kernels_unittest)
endif (Ibis_BUILD_TESTS)
