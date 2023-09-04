#include <iostream>
#include <Python.h>
#include "clean.h"

int clean(int argc, char* argv[]) {
    wchar_t *program = Py_DecodeLocale(argv[0], NULL);
    if (program == NULL) {
        std::cerr << "ibis clean unable to decode program name" << std::endl;
        return 1;
    }

    std::string clean_path = std::string(std::getenv("IBIS")) + "/lib";

    Py_SetProgramName(program);
    Py_Initialize();


    PyRun_SimpleString("import sys");
    std::string import_string = "sys.path.append('" + clean_path + "')";
    PyRun_SimpleString(import_string.c_str());

    PyObject* clean_module = PyImport_ImportModule("clean");
    if (clean_module == NULL) {
        std::cerr << "Failed to import clean.py" << std::endl;
        return 1;
    }
    
    PyObject *py_clean_main = PyObject_GetAttrString(clean_module, "main");
    if (py_clean_main == NULL) {
        std::cerr << "Failed to find main function in prep.py\n";
        Py_DECREF(clean_module);
        return 1;
    }

    PyObject* main_result = PyObject_CallObject(py_clean_main, NULL);
    if (main_result == NULL) {
        std::cerr << "Failed to execute main function in post.py\n"; 
    }
    
    Py_DECREF(py_clean_main);
    Py_DECREF(clean_module);
    
    Py_Finalize();
    return 0;
}
