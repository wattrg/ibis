#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include "prep.h"


int prep(int argc, char* argv[]) {
    wchar_t *program = Py_DecodeLocale(argv[0], NULL);
    if (program == NULL) {
        std::cerr << "ibis prep unable to decode program name" << std::endl;
        return 1;
    }

    if (argc < 3) {
        std::cerr << "Please provide the file to prepare\n";
        return 1;
    }

    std::string prep_path = std::string(std::getenv("IBIS")) + "/lib";

    Py_SetProgramName(program);
    Py_Initialize();

    PyObject* prep_script_name = PyUnicode_FromString(argv[2]);
    if (prep_script_name == NULL) {
        std::cerr << "Failed to interpret the name of the preparation script\n";
        return 1;
    }

    PyRun_SimpleString("import sys");
    std::string import_string = "sys.path.append('" + prep_path + "')";
    PyRun_SimpleString(import_string.c_str());

    PyObject* prep_module = PyImport_ImportModule("prep");
    if (prep_module == NULL) {
        std::cerr << "Failed to import prep.py" << std::endl;
        return 1;
    }
    
    PyObject *py_prep_main = PyObject_GetAttrString(prep_module, "main");
    if (py_prep_main == NULL) {
        std::cerr << "Failed to find main function in prep.py\n";
        Py_DECREF(prep_module);
        return 1;
    }

    PyObject* main_args = PyTuple_New(1);
    PyTuple_SetItem(main_args, 0, prep_script_name);
    PyObject_CallObject(py_prep_main, main_args);
    
    Py_DECREF(py_prep_main);
    Py_DECREF(prep_module);
    Py_DECREF(main_args);
    
    Py_Finalize();
    return 0;
}
