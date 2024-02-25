#ifndef PLOT_H
#define PLOT_H

#include <nlohmann/json.hpp>
#include <CLI/CLI.hpp>
#include <string>
#include <io/io.h>

using json = nlohmann::json;


int plot(FlowFormat format, std::vector<std::string> extras, int argc, char* argv[]);

template <typename T>
void plot_vtk(json directories, std::vector<std::string> extra_vars);

#endif
