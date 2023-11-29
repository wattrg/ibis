#include "vtk.h"

#include "Kokkos_Core_fwd.hpp"
#include "accessor.h"

using array_layout = Kokkos::DefaultExecutionSpace::array_layout;
using host_mem_space = Kokkos::DefaultHostExecutionSpace::memory_space;

int vtk_type_from_elem_type(ElemType type) {
    switch (type) {
        case ElemType::Tri:
            return 5;
        case ElemType::Quad:
            return 9;
        default:
            throw std::runtime_error("Not implemented yet");
    }
}

template <typename T>
void write_scalar_field(std::ofstream& f,
                        const FlowStates<T, array_layout, host_mem_space> fs,
                        std::shared_ptr<ScalarAccessor<T>> accessor,
                        std::string name, std::string type, int num_values) {
    f << "<DataArray type='" << type << "' ";
    f << "NumberOfComponents='1' ";
    f << "Name='" << name << "' ";
    f << "format='ascii'>" << std::endl;

    for (int i = 0; i < num_values; i++) {
        f << accessor->access(fs, i) << std::endl;
    }

    f << "</DataArray>" << std::endl;
}

template <typename T>
void write_vector_field(std::ofstream& f,
                        const Vector3s<T, array_layout, host_mem_space>& vec,
                        std::string name, std::string type, int num_values) {
    f << "<DataArray type='" << type << "' ";
    f << "NumberOfComponents='3' ";
    f << "Name='" << name << "' ";
    f << "format='ascii'>" << std::endl;

    for (int i = 0; i < num_values; i++) {
        f << vec.x(i) << " " << vec.y(i) << " " << vec.z(i) << std::endl;
    }

    f << "</DataArray>" << std::endl;
}

void write_int_view(
    std::ofstream& f,
    const Kokkos::View<int*, array_layout, host_mem_space>& view,
    std::string name, std::string type, bool skip_first = false) {
    f << "<DataArray type='" << type << "' "
      << "NumberOfComponents='1' "
      << "Name='" << name << "' format='ascii'>" << std::endl;
    for (unsigned int i = 0; i < view.extent(0); i++) {
        if (!(skip_first && i == 0)) {
            f << view(i) << std::endl;
        }
    }
    f << "</DataArray>" << std::endl;
}

void write_elem_type(
    std::ofstream& f,
    const Field<ElemType, array_layout, host_mem_space>& types) {
    f << "<DataArray type='Int64' NumberOfComponents='1' Name='types' "
         "format='ascii'>"
      << std::endl;
    for (int i = 0; i < types.size(); i++) {
        int vtk_type = vtk_type_from_elem_type(types(i));
        f << vtk_type << std::endl;
    }
    f << "</DataArray>" << std::endl;
}

template <typename T>
VtkOutput<T>::VtkOutput() {
    m_scalar_accessors = get_accessors<T>();
}

template <typename T>
int VtkOutput<T>::write(const typename FlowStates<T>::mirror_type& fs,
                        const typename GridBlock<T>::mirror_type& grid,
                        std::string plot_dir, std::string time_dir,
                        double time) {
    std::ofstream f(plot_dir + "/" + time_dir + "/" + "/block_0.vtu");
    f << "<VTKFile type='UnstructuredGrid' byte_order='BigEndian'>"
      << std::endl;
    f << "<UnstructuredGrid>" << std::endl;

    // points
    f << "<Piece NumberOfPoints='" << grid.num_vertices() << "' NumberOfCells='"
      << grid.num_cells() << "'>" << std::endl;
    f << "<Points>" << std::endl;
    write_vector_field(f, grid.vertices().positions(), "points", "Float64",
                       grid.num_vertices());
    f << "</Points>" << std::endl;
    f << "<Cells>" << std::endl;
    write_int_view(f, grid.cells().vertex_ids().ids(), "connectivity", "Int64");
    write_int_view(f, grid.cells().vertex_ids().offsets(), "offsets", "Int64",
                   true);
    write_elem_type(f, grid.cells().shapes());
    f << "</Cells>" << std::endl;

    // the cell data
    f << "<CellData>" << std::endl;
    for (auto& key_value : m_scalar_accessors) {
        std::string name = key_value.first;
        std::shared_ptr<ScalarAccessor<T>> accessor = key_value.second;
        write_scalar_field(f, fs, accessor, name, "Float64", grid.num_cells());
    }
    write_vector_field(f, fs.vel, "velocity", "Float64", grid.num_cells());
    f << "</CellData>" << std::endl;

    // close all the data fields
    f << "</Piece>" << std::endl;
    f << "</UnstructuredGrid>" << std::endl;
    f << "</VTKFile>";
    f.close();

    // register that we've written this file
    times_.push_back(time);
    dirs_.push_back(time_dir + "/block_0.vtu");
    return 0;
}

template <typename T>
void VtkOutput<T>::write_coordinating_file(std::string plot_dir) {
    std::ofstream plot_file(plot_dir + "/plot.pvd");
    plot_file << "<?xml version='1.0'?>" << std::endl;
    plot_file
        << "<VTKFile type='Collection' version='0.1' byte_order='LittleEndian'>"
        << std::endl;
    plot_file << "<Collection>" << std::endl;
    for (unsigned int i = 0; i < times_.size(); i++) {
        plot_file << "<DataSet timestep='" << times_[i]
                  << "' group='' part='0' file='" << dirs_[i] << "'/>"
                  << std::endl;
    }
    plot_file << "</Collection>" << std::endl;
    plot_file << "</VTKFile>" << std::endl;
}

template class VtkOutput<double>;
