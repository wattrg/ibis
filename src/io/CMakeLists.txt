add_library(
	IO
	STATIC
	src/io.cpp
	src/native.cpp
	src/vtk.cpp
	src/accessor.cpp
)

target_link_libraries(
	IO
	PUBLIC
	grid
	spdlog::spdlog
)
