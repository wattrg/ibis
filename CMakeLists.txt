cmake_minimum_required(VERSION 3.22.1)

project(aeolus)


add_subdirectory(src/gas)
add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC gas)
