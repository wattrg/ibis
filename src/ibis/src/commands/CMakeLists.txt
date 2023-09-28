find_package(Python COMPONENTS Interpreter Development)

add_library(prep STATIC prep.cpp)
target_link_libraries(prep PRIVATE Python::Python spdlog::spdlog)

add_library(ibis_clean STATIC clean.cpp)
target_link_libraries(ibis_clean PRIVATE Python::Python spdlog::spdlog)

add_library(run STATIC run.cpp)
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
