add_subdirectory(ibis/commands)

add_executable(ibis ibis/main.cpp)

target_link_libraries(
	ibis 
	PUBLIC 
	prep 
	run
	post
  partition
	ibis_clean 
	gas 
	util 
	grid 
	doctest 
	ibis_git_version 
	ibis_version
	runtime_dirs
	spdlog::spdlog
	solver
	IO
	simulation
	CLI11::CLI11
)

if (Ibis_ENABLE_DISTRIBUTED_MEMORY)
	target_compile_definitions(ibis PUBLIC Ibis_ENABLE_DISTRIBUTED_MEMORY)
	target_link_libraries(ibis PUBLIC run_dist)
endif()

set_target_properties(ibis PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

if (Ibis_LINK_FS)
	target_link_libraries(ibis PUBLIC stdc++fs)
endif()
install(TARGETS ibis DESTINATION ${CMAKE_INSTALL_BIN_DIR})
