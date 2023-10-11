#include "vtk.h"

template <typename T>
int VtkOutput<T>::write(const FlowStates<T>& fs, const GridBlock<T>& grid, std::string dir, double time){
    // std::ofstream f(dir + "/block_0.)
    (void) fs;
    (void) grid;
    (void) dir;
    (void) time;
    return 0;
}
template class VtkOutput<double>;
