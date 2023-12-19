add_library(solver STATIC solvers/solver.cpp solvers/solver.h)

target_link_libraries(
	solver 
	PUBLIC 
	Kokkos::kokkos 
	nlohmann_json::nlohmann_json 
	grid 
	gas
	spdlog::spdlog
	IO
)

add_library(runge_kutta STATIC solvers/runge_kutta.cpp)

target_include_directories(solver PUBLIC .)

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
