#include "vtk.h"


template <typename T>
void write_scalar_field(std::ofstream f, T& (*var)(const int), std::string name, std::string type, int num_values) {
    f << "<DataArray type='" << type << "' ";
    f << "NumberOfComponents='1'";
    f << "Name='" << name << "' ";
    f << "format='ascii'>" << std::endl;

    for (int i = 0; i < num_values; i++) {
        f << var(i) << std::endl;
    }

    f << "</DataArray>" << std::endl;
}

template <typename T>
void write_vector_field(std::ofstream& f, 
                        Vector3s<T> vec, 
                        std::string name, 
                        std::string type, 
                        int num_values) 
{
    f << "<DataArray type='" << type << "' ";
    f << "NumberOfComponents='3' ";
    f << "Name='" << name << "' ";
    f << "format='ascii'>" << std::endl;

    for (int i = 0; i < num_values; i++) {
        f << vec.x(i) << " " << vec.y(i) << " " << vec.z(i) << std::endl;
    }

    f << "</DataArray>" << std::endl;
}

template <typename T>
int VtkOutput<T>::write(const FlowStates<T>& fs, 
                        const GridBlock<T>& grid, 
                        std::string dir, 
                        double time)
{
    std::ofstream f(dir + "/block_0.vtu");
    f << "<VTKFile type='UnstructuredGrid' byte_order='BigEndian'>" << std::endl;
    f << "<UnstructuredGrid>" << std::endl;

    // points
    f << "<Points>" << std::endl;
    write_vector_field(f, 
                       grid.vertices().positions(), 
                       "points", 
                       "Float64", 
                       grid.num_cells()); 
    f << "</Points>" << std::endl;
    f << "</UnstructuredGrid>" << std::endl;
    f << "</VTKFile>";
    f.close();
    times_.push_back(time);
    dirs_.push_back(dir);
    return 0;
}
template class VtkOutput<double>;
