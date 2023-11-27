pybind11_add_module(
	python_api
	src/python_api.cpp 
)
set_target_properties(python_api PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)

target_link_libraries(
	python_api
	PRIVATE 
	Kokkos::kokkos
)

install(TARGETS python_api DESTINATION ${CMAKE_INSTALL_LIBDIR})
