find_package(Python COMPONENTS Interpreter Development)

include(${CMAKE_SOURCE_DIR}/cmake/file_embed.cmake)

add_library(prep STATIC prep.cpp)
FileEmbedSetup()
FileEmbedAdd(${CMAKE_SOURCE_DIR}/src/ibis/src/commands/prep.py)
target_link_libraries(prep PRIVATE Python::Python)
target_link_libraries(prep PRIVATE file_embed)
