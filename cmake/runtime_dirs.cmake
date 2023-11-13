# https://jonathanhamberg.com/post/cmake-embedding-git-hash/

SET(CURRENT_LIST_DIR ${CMAKE_CURRENT_LIST_DIR})
SET(pre_configure_dir ${CMAKE_CURRENT_LIST_DIR})
SET(post_configure_dir ${CMAKE_BINARY_DIR}/generated)

SET(pre_configure_file ${pre_configure_dir}/runtime_dirs.cpp.in)
SET(post_configure_file ${post_configure_dir}/runtime_dirs.cpp)

FUNCTION(configure_runtime_dirs)
  IF(NOT EXISTS ${post_configure_dir}/runtime_dirs.h)
    FILE(
      COPY ${pre_configure_dir}/runtime_dirs.h
      DESTINATION ${post_configure_dir})
  ENDIF()

  configure_file(${pre_configure_file} ${post_configure_file} @ONLY)
  # message(STATUS "Configured git information in ${post_configure_file}")
ENDFUNCTION()

FUNCTION(setup_runtime_dirs)
  add_library(runtime_dirs ${CMAKE_BINARY_DIR}/generated/runtime_dirs.cpp)
  target_include_directories(runtime_dirs PUBLIC ${CMAKE_BINARY_DIR}/generated)
  target_compile_features(ibis_git_version PRIVATE cxx_raw_string_literals)

  configure_runtime_dirs()
ENDFUNCTION()
