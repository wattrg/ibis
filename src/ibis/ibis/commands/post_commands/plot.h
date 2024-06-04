#ifndef PLOT_H
#define PLOT_H

#include <io/io.h>

#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>
#include <string>

using json = nlohmann::json;

int plot(FlowFormat format, std::vector<std::string> extras, int argc, char* argv[]);

template <typename T, bool binary>
void plot_vtk(json directories, std::vector<std::string> extra_vars);

#endif
