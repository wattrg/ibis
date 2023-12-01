#ifndef PLOT_VTK_H
#define PLOT_VTK_H

#include <nlohmann/json.hpp>
#include <string>

using json = nlohmann::json;

template <typename T>
void plot_vtk(json directories);

#endif
