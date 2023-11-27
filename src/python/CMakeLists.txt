pybind11_add_module(
	python_api
	src/python_api.cpp 
)
set_property(TARGET python_api PROPERTY CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

target_link_libraries(
	python_api
	PRIVATE 
	Kokkos::kokkos
)

install(TARGETS python_api DESTINATION ${CMAKE_INSTALL_LIBDIR})
