# cmake/PythonVenv.cmake
# Creates a Python venv, installs packages, and handles installation.
#
# Usage:
#   setup_python_venv(
#     NAME        <target-name>          # logical name, e.g. "myapp_venv"
#     VENV_DIR    <path>                 # where to create the venv (build tree)
#     REQUIREMENTS <requirements.txt>   # path to requirements file (optional)
#     INSTALL_DIR <relative-install-path> # e.g. "lib/myapp/venv"
#     CPP_DEFINE  <DEFINE_NAME>          # C++ macro name, e.g. "VENV_PATH"
#   )

function(setup_python_venv)
  cmake_parse_arguments(
    VENV
    ""
    "NAME;DIR;REQUIREMENTS;INSTALL_DIR;CPP_DEFINE"
    ""
    ${ARGN}
  )

  # ── Validate ──────────────────────────────────────────────────────────────
  # foreach(required NAME DIR REQUIREMENTS INSTALL_DIR CPP_DEFINE)
  #   if(NOT VENV_${required})
  #     message(FATAL_ERROR "setup_python_venv: ${required} is required")
  #   endif()
  # endforeach()

  # ── Find Python ───────────────────────────────────────────────────────────
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  message(STATUS "[venv] Using Python: ${Python3_EXECUTABLE}")

  # ── Paths ─────────────────────────────────────────────────────────────────
  set(VENV_STAMP "${VENV_DIR}/.cmake_venv_stamp")

  if(WIN32)
    set(VENV_PYTHON "${VENV_DIR}/Scripts/python.exe")
    set(VENV_PIP    "${VENV_DIR}/Scripts/pip.exe")
  else()
    set(VENV_PYTHON "${VENV_DIR}/bin/python")
    set(VENV_PIP    "${VENV_DIR}/bin/pip")
  endif()

  # ── CONFIGURE TIME: create the venv ───────────────────────────────────────
  if(NOT EXISTS "${VENV_DIR}/pyvenv.cfg")
    message(STATUS "[venv] Creating virtual environment at ${VENV_DIR}")
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -m venv "${VENV_DIR}"
      RESULT_VARIABLE result
    )
    if(result)
      message(FATAL_ERROR "[venv] Failed to create venv (exit code ${result})")
    endif()
  else()
    message(STATUS "[venv] Virtual environment already exists at ${VENV_DIR}")
  endif()

  # ── BUILD TIME: install packages ──────────────────────────────────────────
  set(pip_install_cmd "${VENV_PIP}" install --upgrade pip)
  set(install_commands COMMAND "${VENV_PIP}" install --upgrade pip)

  if(VENV_REQUIREMENTS)
    list(APPEND install_commands
      COMMAND "${VENV_PIP}" install -r "${VENV_REQUIREMENTS}"
    )
  endif()

  foreach(pkg IN LISTS VENV_PACKAGES)
    list(APPEND install_commands
      COMMAND "${VENV_PIP}" install "${pkg}"
    )
  endforeach()

  add_custom_command(
    OUTPUT  "${VENV_STAMP}"
    ${install_commands}
    COMMAND "${CMAKE_COMMAND}" -E touch "${VENV_STAMP}"
    COMMENT "[venv] Installing Python packages into ${VENV_DIR}"
    VERBATIM
  )

  add_custom_target("${VENV_NAME}_setup" ALL
    DEPENDS "${VENV_STAMP}"
  )

  # ── INSTALL TIME: copy venv ────────────────────────────────────────────────
  # The full installed path (for the C++ define)
  set(INSTALLED_VENV_PATH "${CMAKE_INSTALL_PREFIX}/${VENV_INSTALL_DIR}")

  install(
    DIRECTORY   "${VENV_DIR}/"
    DESTINATION "${VENV_INSTALL_DIR}"
    USE_SOURCE_PERMISSIONS
  )

  # Fix up shebangs/absolute paths in the venv after copying.
  # Python's venv uses absolute paths; we relocate them with a post-install script.
  install(CODE "
    message(STATUS \"[venv] Relinking venv at \${CMAKE_INSTALL_PREFIX}/${VENV_INSTALL_DIR}\")
    execute_process(
      COMMAND \"${Python3_EXECUTABLE}\" -m venv
              --upgrade
              \"\${CMAKE_INSTALL_PREFIX}/${VENV_INSTALL_DIR}\"
      RESULT_VARIABLE result
    )
    if(result)
      message(WARNING \"[venv] venv relink exited with: \${result}\")
    endif()
  ")

  # ── C++ compile definition ─────────────────────────────────────────────────
  # Expose to all targets added after this call via a cached string.
  # Callers can also use target_compile_definitions() with the variable.
  set(${VENV_CPP_DEFINE}_PATH "${INSTALLED_VENV_PATH}"
    CACHE INTERNAL "Installed venv path for C++ define ${VENV_CPP_DEFINE}"
  )

  # Helper target property so consumers can do:
  #   target_link_libraries(myapp PRIVATE ${VENV_NAME}_venv_iface)
  add_library("${VENV_NAME}_venv_iface" INTERFACE)
  target_compile_definitions("${VENV_NAME}_venv_iface" INTERFACE
    ${VENV_CPP_DEFINE}="${INSTALLED_VENV_PATH}"
  )

  message(STATUS "[venv] C++ define: ${VENV_CPP_DEFINE}=\"${INSTALLED_VENV_PATH}\"")
endfunction()
