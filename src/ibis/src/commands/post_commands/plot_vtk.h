#ifndef PLOT_VTK_H
#define PLOT_VTK_H

#include <string>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

template <typename T>
void plot_vtk(json directories);

#endif
