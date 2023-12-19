add_library(
	gas
	STATIC
	gas/gas_state.cpp
	gas/gas_model.cpp
	gas/flow_state.cpp
)

target_link_libraries(
	gas
	PUBLIC
	Kokkos::kokkos 
	nlohmann_json::nlohmann_json
	util
)

target_include_directories(
	gas
	PUBLIC
	.
)
