add_library(
	IO
	STATIC
	src/io.cpp
	src/native.cpp
)

target_link_libraries(
	IO
	PUBLIC
	grid
	spdlog::spdlog
)
