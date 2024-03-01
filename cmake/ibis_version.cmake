# https://jonathanhamberg.com/post/cmake-embedding-git-hash/

SET(CURRENT_LIST_DIR ${CMAKE_CURRENT_LIST_DIR})
SET(pre_configure_dir ${CMAKE_CURRENT_LIST_DIR})
SET(post_configure_dir ${CMAKE_BINARY_DIR}/generated)

SET(pre_configure_file ${pre_configure_dir}/ibis_version.cpp.in)
SET(post_configure_file ${post_configure_dir}/ibis_version.cpp)

FUNCTION(configure_version)
  IF(NOT EXISTS ${post_configure_dir}/ibis_version.h)
    FILE(
      COPY ${pre_configure_dir}/ibis_version.h
      DESTINATION ${post_configure_dir})
  ENDIF()

  configure_file(${pre_configure_file} ${post_configure_file} @ONLY)
  # message(STATUS "Configured git information in ${post_configure_file}")
ENDFUNCTION()

FUNCTION(setup_ibis_version)
  add_library(ibis_version ${CMAKE_BINARY_DIR}/generated/ibis_version.cpp)
  target_include_directories(ibis_version PUBLIC ${CMAKE_BINARY_DIR}/generated)
  # target_compile_features(ibis_git_version PRIVATE cxx_raw_string_literals)

  configure_version()
ENDFUNCTION()
