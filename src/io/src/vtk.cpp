#include "vtk.h"

template <typename T>
int write_vtk(const FlowStates<T>& fs, const GridBlock<T>& grid, std::string dir){
    (void) fs;
    (void) grid;
    (void) dir;
    return 0;
}
template int write_vtk<double>(const FlowStates<double>&, const GridBlock<double>&, std::string);
