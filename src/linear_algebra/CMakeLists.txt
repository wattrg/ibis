add_library(
    linear_algebra
    STATIC
    linear_algebra/gmres.cpp 
    linear_algebra/dense_linear_algebra.cpp
    linear_algebra/ilu.cpp
    linear_algebra/reduced_basis_preconditioner.cpp
)
target_link_libraries(
    linear_algebra 
    PUBLIC 
    Kokkos::kokkos 
    doctest 
    simulation
    util 
    finite_volume 
    doctest
    ilu_kokkos_kernels
    triangular_solver_kokkos_kernels
)
target_include_directories(linear_algebra PUBLIC .)
target_compile_definitions(linear_algebra PRIVATE DOCTEST_CONFIG_DISABLE)


if (Ibis_BUILD_TESTS)
    add_executable(
        linear_algebra_unittest 
        test/unittest.cpp
        linear_algebra/dense_linear_algebra.cpp
        linear_algebra/gmres.cpp
    )

    target_link_libraries(
	      linear_algebra_unittest 	
	      PRIVATE
	      Kokkos::kokkos
	      doctest
	      util
	      simulation
    )

    target_include_directories(linear_algebra_unittest PRIVATE .)

    add_test(NAME linear_algebra_unittest COMMAND linear_algebra_unittest)
endif(Ibis_BUILD_TESTS)
