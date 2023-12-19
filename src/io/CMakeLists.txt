add_library(
	IO
	STATIC
	io/io.cpp
	io/native.cpp
	io/vtk.cpp
	io/accessor.cpp
)

target_link_libraries(
	IO
	PUBLIC
	grid
	spdlog::spdlog
	gas
)

target_include_directories(
	IO
	PUBLIC
	.
)
