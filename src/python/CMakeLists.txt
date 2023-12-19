pybind11_add_module(
	python_api
	NO_EXTRAS
	python/python_api.cpp 
)
set_target_properties(python_api PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)

target_link_libraries(
	python_api
	PRIVATE 
	Kokkos::kokkos
	doctest
	spdlog::spdlog
	finite_volume
	gas
)

install(TARGETS python_api DESTINATION ${CMAKE_INSTALL_LIBDIR})
