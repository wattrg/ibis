add_library(util STATIC vector3.cpp)

add_executable(util_unittest unittest.cpp vector3.cpp)
target_link_libraries(util_unittest doctest::doctest)
