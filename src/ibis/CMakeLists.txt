add_subdirectory(src/commands)

add_executable(ibis src/main.cpp)
target_include_directories(ibis PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(
	ibis 
	PUBLIC 
	prep 
	run 
	post
	ibis_clean 
	gas 
	util 
	grid 
	doctest 
	ibis_git_version 
	runtime_dirs
	spdlog::spdlog
	solver
	stdc++fs
)
install(TARGETS ibis DESTINATION ${CMAKE_INSTALL_BIN_DIR})
