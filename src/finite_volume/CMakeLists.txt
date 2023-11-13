add_library(
	finite_volume 
	STATIC 
	src/finite_volume.cpp 
	src/flux_calc.cpp 
	src/flux_calculators/hanel.cpp
	src/flux_calculators/ausmdv.cpp
	src/primative_conserved_conversion.cpp
	src/conserved_quantities.cpp
	src/boundaries/boundary.cpp
)

target_link_libraries(
	finite_volume 
	PUBLIC
	Kokkos::kokkos 
	doctest 
	util
	nlohmann_json::nlohmann_json
	spdlog::spdlog
)

if (Ibis_BUILD_TESTS)
    add_executable(
    	finite_volume_unittest 
    	test/unittest.cpp 
    	src/flux_calc.cpp 
    	src/flux_calculators/hanel.cpp
    	src/flux_calculators/ausmdv.cpp
    )
    # target_compile_options(finite_volume_unittest -g)
    target_link_libraries(
    	finite_volume_unittest 
    	PRIVATE 
    	util
    	doctest
    	Kokkos::kokkos
    	nlohmann_json::nlohmann_json
    	spdlog::spdlog
    )
    
    add_test(NAME finite_volume_unittest COMMAND finite_volume_unittest)
endif(Ibis_BUILD_TESTS)
