find_package(Python COMPONENTS Interpreter Development)

add_library(prep STATIC prep.cpp)
target_link_libraries(prep PRIVATE Python::Python spdlog::spdlog runtime_dirs)

add_library(ibis_clean STATIC clean.cpp)
target_link_libraries(ibis_clean PRIVATE Python::Python spdlog::spdlog runtime_dirs)

add_library(run STATIC run.cpp ../config.cpp)
target_link_libraries(
        run PUBLIC 
        Kokkos::kokkos 
        doctest 
        grid 
        runge_kutta 
        solver 
        nlohmann_json::nlohmann_json
        gas
        spdlog::spdlog
        runtime_dirs
)


add_library(post STATIC post.cpp post_commands/plot_vtk.cpp ../config.cpp)
target_link_libraries(
        post PUBLIC
        Python::Python 
        spdlog::spdlog 
        nlohmann_json::nlohmann_json
        Kokkos::kokkos
        gas
        grid
        runtime_dirs
)


add_custom_command(
	TARGET prep POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_SOURCE_DIR}/src/ibis/src/commands/prep.py $<TARGET_FILE_DIR:prep>
        ${CMAKE_SOURCE_DIR}/src/ibis/src/commands/ibis_py_utils.py $<TARGET_FILE_DIR:prep>
)

add_custom_command(
	TARGET ibis_clean POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_SOURCE_DIR}/src/ibis/src/commands/clean.py $<TARGET_FILE_DIR:ibis_clean>
)

install(FILES prep.py 
        DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES ibis_py_utils.py 
        DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES clean.py
        DESTINATION ${CMAKE_INSTALL_LIBDIR})
