add_library(runge_kutta STATIC runge_kutta.cpp)
target_link_libraries(
	runge_kutta 
	PUBLIC 
	Kokkos::kokkos 
	doctest 
	util 
	grid 
	nlohmann_json::nlohmann_json
	finite_volume
	solver
	spdlog::spdlog
	IO
)
