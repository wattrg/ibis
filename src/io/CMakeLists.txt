add_library(
	IO
	STATIC
	src/io.cpp
)

target_link_libraries(
	IO
	PUBLIC
	grid
	spdlog::spdlog
)
