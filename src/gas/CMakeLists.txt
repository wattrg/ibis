add_library(
	gas
	STATIC
	src/gas_state.cpp
	src/gas_model.cpp
	src/flow_state.cpp
)

target_link_libraries(
	gas
	PUBLIC
	Kokkos::kokkos 
	nlohmann_json::nlohmann_json
	util
)
