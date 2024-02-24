#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <ibis/commands/prep/prep.h>
#include <runtime_dirs.h>
#include <spdlog/spdlog.h>

#include <cstdlib>
#include <fstream>
#include <iostream>

void setup_prep_cli(CLI::App& ibis) {
    ibis.add_subcommand("prep", "prepare the simulation");
}

int prep(int argc, char* argv[]) {
    (void)argc;
    (void)argv;
    Py_Initialize();

    PyObject* prep_script_name = PyUnicode_FromString("job.py");
    if (prep_script_name == NULL) {
        spdlog::error("Failed to interpret the name of the preparation script");
        Py_Finalize();
        return 1;
    }

    PyObject* res_dir = PyUnicode_FromString(Ibis::RES_DIR.c_str());
    if (res_dir == NULL) {
        spdlog::error("Failed to set library directory");
        Py_Finalize();
        return 1;
    }

    PyRun_SimpleString("import sys");
    std::string import_string = "sys.path.append('" + Ibis::LIB_DIR + "')";
    PyRun_SimpleString(import_string.c_str());

    PyObject* prep_module = PyImport_ImportModule("prep");
    if (prep_module == NULL) {
        PyErr_Print();
        Py_Finalize();
        return 1;
    }

    PyObject* py_prep_main = PyObject_GetAttrString(prep_module, "main");
    if (py_prep_main == NULL) {
        PyErr_Print();
        Py_DECREF(prep_module);
        Py_Finalize();
        return 1;
    }

    PyObject* main_args = PyTuple_New(2);
    PyTuple_SetItem(main_args, 0, prep_script_name);
    PyTuple_SetItem(main_args, 1, res_dir);
    PyObject* py_main_result = PyObject_CallObject(py_prep_main, main_args);
    if (py_main_result == NULL) {
        PyErr_Print();
        Py_DECREF(prep_module);
        Py_DECREF(py_prep_main);
        Py_DECREF(main_args);
        Py_Finalize();
        return 1;
    }

    Py_DECREF(py_prep_main);
    Py_DECREF(prep_module);
    Py_DECREF(main_args);
    Py_Finalize();

    spdlog::info("Preparation complete");
    return 0;
}
