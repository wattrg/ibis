#include <Python.h>
#include <ibis/commands/clean/clean.h>
#include <runtime_dirs.h>
#include <spdlog/spdlog.h>

#include <iostream>

int clean(int argc, char* argv[]) {
    (void)argc;
    (void)argv;
    Py_Initialize();

    PyObject* res_dir = PyUnicode_FromString(Ibis::RES_DIR.c_str());
    if (res_dir == NULL) {
        spdlog::error("Failed to set library directory");
        Py_Finalize();
        return 1;
    }

    PyRun_SimpleString("import sys");
    std::string import_string = "sys.path.append('" + Ibis::LIB_DIR + "')";
    PyRun_SimpleString(import_string.c_str());

    PyObject* clean_module = PyImport_ImportModule("clean");
    if (clean_module == NULL) {
        PyErr_Print();
        Py_Finalize();
        return 1;
    }

    PyObject* py_clean_main = PyObject_GetAttrString(clean_module, "main");
    if (py_clean_main == NULL) {
        PyErr_Print();
        Py_DECREF(clean_module);
        Py_Finalize();
        return 1;
    }

    PyObject* main_args = PyTuple_New(1);
    PyTuple_SetItem(main_args, 0, res_dir);

    PyObject* main_result = PyObject_CallObject(py_clean_main, main_args);
    if (main_result == NULL) {
        PyErr_Print();
    }

    Py_DECREF(py_clean_main);
    Py_DECREF(clean_module);

    Py_Finalize();

    spdlog::info("Cleaned directory");
    return 0;
}
