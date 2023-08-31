find_package(Python COMPONENTS Interpreter Development)

add_library(prep STATIC prep.cpp)
target_link_libraries(prep PRIVATE Python::Python)

add_custom_command(
	TARGET prep POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_SOURCE_DIR}/src/ibis/src/commands/prep.py $<TARGET_FILE_DIR:prep>
)
