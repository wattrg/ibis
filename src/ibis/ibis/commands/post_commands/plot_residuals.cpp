#include <Python.h>
#include <ibis/commands/post_commands/plot_residuals.h>
#include <runtime_dirs.h>
#include <spdlog/spdlog.h>

int plot_residuals() {
    Py_Initialize();

    PyObject* prep_script_name = PyUnicode_FromString("plot_residuals.py");
    if (prep_script_name == NULL) {
        spdlog::error("Failed to interpret the name of the residual plotting script");
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

    PyObject* plot_residuals_module = PyImport_ImportModule("plot_residuals");
    if (plot_residuals_module == NULL) {
        PyErr_Print();
        Py_Finalize();
        return 1;
    }

    PyObject* py_plot_residuals_main =
        PyObject_GetAttrString(plot_residuals_module, "main");
    if (py_plot_residuals_main == NULL) {
        PyErr_Print();
        Py_DECREF(plot_residuals_module);
        Py_Finalize();
        return 1;
    }

    PyObject* py_main_result = PyObject_CallObject(py_plot_residuals_main, NULL);
    if (py_main_result == NULL) {
        PyErr_Print();
        Py_DECREF(plot_residuals_module);
        Py_DECREF(py_plot_residuals_main);
        Py_Finalize();
        return 1;
    }

    Py_DECREF(py_plot_residuals_main);
    Py_DECREF(plot_residuals_module);
    Py_Finalize();

    return 0;
}
