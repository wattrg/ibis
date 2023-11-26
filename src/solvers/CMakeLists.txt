add_library(
	solver 
	STATIC 
	src/solver.cpp
	src/units.cpp
)
target_link_libraries(
	solver 
	PUBLIC 
	Kokkos::kokkos 
	nlohmann_json::nlohmann_json 
	grid 
	gas
	spdlog::spdlog
)

add_subdirectory(src/runge_kutta)
