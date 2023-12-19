add_library(solver STATIC src/solver.cpp src/solver.h)
# target_include_directories(solver PUBLIC ${CMAKE_SOURCE_DIR}/extern/json/single_include)
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

add_library(runge_kutta STATIC src/runge_kutta.cpp)

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
