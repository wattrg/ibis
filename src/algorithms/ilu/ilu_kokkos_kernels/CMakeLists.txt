add_library(
  ilu_kokkos_kernels
  STATIC
  ilu_kokkos_kernels/ilu_kokkos_kernels.cpp
)

target_include_directories(
  ilu_kokkos_kernels
  PUBLIC
  .
)

target_link_libraries(
  ilu_kokkos_kernels
  PUBLIC
  linear_algebra
  Kokkos::kokkos
  KokkosKernels::kokkoskernels
  doctest
  ilu

)

if (Ibis_BUILD_TESTS)
  add_executable(
    ilu_kokkos_kernels_unittest
    test/unittest.cpp
    ilu_kokkos_kernels/ilu_kokkos_kernels.cpp
  )

  target_link_libraries(
    ilu_kokkos_kernels_unittest
    PRIVATE
    util
    ilu_kokkos_kernels
  )

  add_test(NAME ilu_kokkos_kernels COMMAND ilu_kokkos_kernels_unittest)
endif (Ibis_BUILD_TESTS)
