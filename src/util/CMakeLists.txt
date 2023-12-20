add_library(util STATIC util/vector3.cpp util/id.cpp util/geom.cpp)
target_link_libraries(util PUBLIC Kokkos::kokkos doctest)
target_include_directories(util PUBLIC .)


if (Ibis_BUILD_TESTS)
    add_executable(util_unittest 
		   test/unittest.cpp 
		   util/vector3.cpp 
		   util/id.cpp
		   util/geom.cpp)
    # target_compile_options(util_unittest -g)
    target_link_libraries(util_unittest 	
	PRIVATE Kokkos::kokkos doctest)
    add_test(NAME util_unittest COMMAND util_unittest)
endif(Ibis_BUILD_TESTS)
