# https://jonathanhamberg.com/post/cmake-embedding-git-hash/

find_package(Git QUIET)

SET(CURRENT_LIST_DIR ${CMAKE_CURRENT_LIST_DIR})
SET(pre_configure_dir ${CMAKE_CURRENT_LIST_DIR})
SET(post_configure_dir ${CMAKE_BINARY_DIR}/generated)

SET(pre_configure_file ${pre_configure_dir}/ibis_version_info.cpp.in)
SET(post_configure_file ${post_configure_dir}/ibis_version_info.cpp)

FUNCTION(check_git_write git_hash git_clean_status)
  FILE(
    WRITE
    ${CMAKE_BINARY_DIR}/git-state.txt
    "${git_hash}-${git_clean_status}")
ENDFUNCTION()

FUNCTION(check_git_read git_hash)
  IF(EXISTS ${CMAKE_BINARY_DIR}/git-state.txt)
    FILE(STRINGS ${CMAKE_BINARY_DIR}/git-state.txt CONTENT)
    LIST(GET CONTENT 0 var)

    message(DEBUG "Cached Git hash: ${var}")
    SET(${git_hash} ${var} PARENT_SCOPE)
  else()
    SET(${git_hash} "INVALID" PARENT_SCOPE)
  ENDIF()
ENDFUNCTION()

FUNCTION(check_git_version)
  IF(NOT EXISTS ${post_configure_dir}/ibis_version_info.h)
    FILE(
      COPY ${pre_configure_dir}/ibis_version_info.h
      DESTINATION ${post_configure_dir})
  ENDIF()

  # IF(NOT Git_FOUND OR NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git)
  #   configure_file(${pre_configure_file} ${post_configure_file} @ONLY)
  #   return()
  # ENDIF()

  # Get the current working branch
  execute_process(
    COMMAND git rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  # Get the latest commit description
  execute_process(
    COMMAND git show -s --format=%s
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_DESCRIPTION
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  # Get the latest commit date
  set(DATE_FORMAT "format-local:%a %d %b %Y %H:%M:%S %Z")
  execute_process(
    COMMAND git show -s --format=%cd --date=${DATE_FORMAT}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_DATE
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  # Get the build date
  execute_process(
    COMMAND date
    OUTPUT_VARIABLE BUILD_DATE
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  # Check if repo is dirty / clean
  execute_process(
    COMMAND git diff-index --quiet HEAD --
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    RESULT_VARIABLE IS_DIRTY
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  IF(IS_DIRTY EQUAL 0)
    SET(GIT_CLEAN_STATUS "clean")
  else()
    SET(GIT_CLEAN_STATUS "dirty")
  ENDIF()

  # Get the latest abbreviated commit hash of the working branch
  execute_process(
    COMMAND git log -1 --format=%h
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  check_git_read(GIT_HASH_CACHE)

  IF(NOT EXISTS ${post_configure_dir})
    file(MAKE_DIRECTORY ${post_configure_dir})
  ENDIF()

  check_git_write(${GIT_COMMIT_HASH} ${GIT_CLEAN_STATUS})

  configure_file(${pre_configure_file} ${post_configure_file} @ONLY)
  message(STATUS "Configured git information in ${post_configure_file}")
ENDFUNCTION()

FUNCTION(check_git_setup)
  add_custom_target(
    IbisAlwaysCheckGit COMMAND ${CMAKE_COMMAND}
    -DRUN_CHECK_GIT_VERSION=1
    -Dpre_configure_dir=${pre_configure_dir}
    -Dpost_configure_file=${post_configure_dir}
    -DGIT_HASH_CACHE=${GIT_HASH_CACHE}
    -P ${CURRENT_LIST_DIR}/build_env_info.cmake
    BYPRODUCTS ${post_configure_file})

  add_library(ibis_git_version ${CMAKE_BINARY_DIR}/generated/ibis_version_info.cpp)
  target_include_directories(ibis_git_version PUBLIC ${CMAKE_BINARY_DIR}/generated)
  target_compile_features(impl_git_version PRIVATE cxx_raw_string_literals)
  add_dependencies(ibis_git_version IbisAlwaysCheckGit)

  check_git_version()
ENDFUNCTION()

# This is used to run this function from an external cmake process.
IF(RUN_CHECK_GIT_VERSION)
  check_git_version()
ENDIF()
