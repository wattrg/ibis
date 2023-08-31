#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <iostream>
#include <sstream>
#include "prep.h"
#include "prep_py.h"


std::string bytes_to_string(uint8_t str[], unsigned size) {
    std::stringstream result;
    for (unsigned i = 0; i < size; i++) {
        result << str[i];
    }
    return result.str();
}

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

    std::string prep_script = bytes_to_string(prep_py_data, prep_py_size);

    Py_SetProgramName(program);
    Py_Initialize();

    PyObject* prep_script_name = PyUnicode_FromString(argv[2]);
    if (prep_script_name == NULL) {
        std::cerr << "Failed to interpret the name of the preparation script\n";
        return 1;
    }

    // Define code in the newly created module
    PyObject* py_compiled_prep = Py_CompileString(prep_script.c_str(), "", Py_file_input);
    if (py_compiled_prep == NULL) {
        std::cerr << "Failed to compile prep.py\n";
        return 1;
    }

    PyObject *prep_module = PyImport_ExecCodeModule("prep", py_compiled_prep);
    
    if (prep_module == NULL) {
        std::cerr << "Failed to import prep.py" << std::endl;
        Py_DECREF(py_compiled_prep);
        return 1;
    }

    
    PyObject *py_prep_main = PyObject_GetAttrString(prep_module, "main");
    if (py_prep_main == NULL) {
        std::cerr << "Failed to find main function in prep.py\n";
        Py_DECREF(prep_module);
        Py_DECREF(py_compiled_prep);
        return 1;
    }

    PyObject* main_args = PyTuple_New(1);
    PyTuple_SetItem(main_args, 0, prep_script_name);
    PyObject_CallObject(py_prep_main, main_args);
    
    Py_DECREF(py_prep_main);
    Py_DECREF(prep_module);
    Py_DECREF(main_args);
    Py_DECREF(py_compiled_prep);
    
    Py_Finalize();
    return 0;
}
