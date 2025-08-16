add_library(
	solver
	STATIC
	solvers/solver.cpp
	solvers/solver.h
	solvers/cfl.cpp
)


add_library(runge_kutta STATIC solvers/runge_kutta.cpp)
add_library(jfnk STATIC solvers/jfnk.cpp)
add_library(steady_state STATIC solvers/steady_state.cpp)

target_include_directories(solver PUBLIC .)

target_link_libraries(
	runge_kutta 
	PUBLIC 
	doctest 
	util 
	grid 
	nlohmann_json::nlohmann_json
	finite_volume
	solver
	spdlog::spdlog
	IO
	parallel
)
target_link_libraries(
	jfnk 
	PUBLIC 
	doctest 
	util 
	grid 
	nlohmann_json::nlohmann_json
	finite_volume
	solver
	spdlog::spdlog
	IO
)
target_link_libraries(
	steady_state 
	PUBLIC 
	doctest 
	util 
	grid 
	nlohmann_json::nlohmann_json
	finite_volume
	solver
	spdlog::spdlog
	IO
	jfnk
)
target_link_libraries(
	solver 
	PUBLIC 
	nlohmann_json::nlohmann_json 
	linear_algebra
	grid 
	gas
	spdlog::spdlog
	IO
	simulation
	steady_state
	runge_kutta
)


if (Ibis_BUILD_TESTS)
	add_executable(
		solver_unittest
		test/unittest.cpp
		solvers/cfl.cpp
	)
	target_link_libraries(
		solver_unittest 
		PRIVATE
		nlohmann_json::nlohmann_json 
		grid 
		gas
		spdlog::spdlog
		IO
	)
	add_test(NAME solver_unittest COMMAND solver_unittest)
endif(Ibis_BUILD_TESTS)
