#include "vtk.h"

int vtk_type_from_elem_type(ElemType type) {
    switch (type) {
        case ElemType::Quad:
            return 9;
        default:
            throw std::runtime_error("Not implemented yet");
    }
}

template <typename T>
class ScalarAccessor{
public:
    virtual const T access(const FlowStates<T>& fs, const int i) = 0;
};

template <typename T>
class VectorAccessor{
public:
    virtual const T access_x(const FlowStates<T>& fs, const int i) = 0;
    virtual const T access_y(const FlowStates<T>& fs, const int i) = 0;
    virtual const T access_z(const FlowStates<T>& fs, const int i) = 0;
};

template <typename T>
class PressureAccess : public ScalarAccessor<T>{
    const T access (const FlowStates<T>& fs, const int i) {
        return fs.gas.pressure(i);
    }
};

template <typename T>
class TemperatureAccess : public ScalarAccessor<T>{
    const T access (const FlowStates<T>& fs, const int i) {
        return fs.gas.temp(i);
    }
};

template <typename T>
class DensityAccess : public ScalarAccessor<T>{
    const T access (const FlowStates<T>& fs, const int i) {
        return fs.gas.rho(i);
    }
};

template <typename T>
class InternalEnergyAccess : public ScalarAccessor<T> {
    const T access (const FlowStates<T>& fs, const int i) {
        return fs.gas.energy(i);
    }
};

template <typename T>
class SpeedOfSoundAccess : public ScalarAccessor<T> {
    const T access (const FlowStates<T>& fs, const int i){
        return Kokkos::sqrt(1.4 * 287.0 * fs.gas.temp(i));
    }
};

template <typename T>
class MachNumberAccess : public ScalarAccessor<T> {
    const T access (const FlowStates<T>& fs, const int i){
        T a = Kokkos::sqrt(1.4 * 287.0 * fs.gas.temp(i));
        T vx = fs.vel.x(i);
        T vy = fs.vel.y(i);
        T vz = fs.vel.z(i);
        T v_mag = Kokkos::sqrt(vx*vx + vy*vy + vz*vz);
        return v_mag / a;
    }
};

template <typename T>
class VelocityAccess : public VectorAccessor<T> {
    const T access_x(const FlowStates<T>& fs, const int i) {
        fs.vel.x(i);
    }
    const T access_y(const FlowStates<T>& fs, const int i) {
        fs.vel.y(i);
    }
    const T access_z(const FlowStates<T>& fs, const int i) {
        fs.vel.z(i);
    }
};

template <typename T>
void write_scalar_field(std::ofstream& f, 
                        const FlowStates<T> fs, 
                        std::unique_ptr<ScalarAccessor<T>>& accessor, 
                        std::string name, 
                        std::string type, 
                        int num_values) 
{
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
                        const Vector3s<T>& vec, 
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

void write_int_view(std::ofstream& f, const Kokkos::View<int*>& view, std::string name, std::string type, bool skip_first=false) {
    f << "<DataArray type='" << type << "' " << "NumberOfComponents='1' " << "Name='" << name << "' format='ascii'>" << std::endl;
    for (unsigned int i = 0; i < view.extent(0); i++){
        if (!(skip_first && i == 0)){
            f << view(i) << std::endl;
        }
    } 
    f << "</DataArray>" << std::endl;
}

void write_elem_type(std::ofstream& f, const Field<ElemType>& types){
    f << "<DataArray type='Int64' NumberOfComponents='1' Name='types' format='ascii'>" 
        << std::endl;
    for (int i = 0; i < types.size(); i++) {
        int vtk_type = vtk_type_from_elem_type(types(i));
        f << vtk_type << std::endl;
    }
    f << "</DataArray>" << std::endl;
} 

template <typename T>
int VtkOutput<T>::write(const FlowStates<T>& fs, 
                        const GridBlock<T>& grid, 
                        std::string plot_dir,
                        std::string time_dir, 
                        double time)
{
    std::ofstream f(plot_dir + "/" + time_dir + "/" + "/block_0.vtu");
    f << "<VTKFile type='UnstructuredGrid' byte_order='BigEndian'>" << std::endl;
    f << "<UnstructuredGrid>" << std::endl;

    // points
    f << "<Piece NumberOfPoints='" 
      << grid.num_vertices() 
      << "' NumberOfCells='" 
      << grid.num_cells() 
      << "'>" << std::endl; 
    f << "<Points>" << std::endl;
    write_vector_field(f, 
                       grid.vertices().positions(), 
                       "points", 
                       "Float64", 
                       grid.num_vertices()); 
    f << "</Points>" << std::endl;
    f << "<Cells>" << std::endl;
    write_int_view(f, grid.cells().vertex_ids().ids(), "connectivity", "Int64");
    write_int_view(f, grid.cells().vertex_ids().offsets(), "offsets", "Int64", true);
    write_elem_type(f, grid.cells().shapes());
    f << "</Cells>" << std::endl;
    f << "<CellData>" << std::endl;
    std::unique_ptr<ScalarAccessor<T>> pressure = std::unique_ptr<ScalarAccessor<T>>(new PressureAccess<T>()); 
    std::unique_ptr<ScalarAccessor<T>> temp = std::unique_ptr<ScalarAccessor<T>>(new TemperatureAccess<T>()); 
    std::unique_ptr<ScalarAccessor<T>> density = std::unique_ptr<ScalarAccessor<T>>(new DensityAccess<T>()); 
    std::unique_ptr<ScalarAccessor<T>> energy = std::unique_ptr<ScalarAccessor<T>>(new InternalEnergyAccess<T>()); 
    std::unique_ptr<ScalarAccessor<T>> a = std::unique_ptr<ScalarAccessor<T>>(new SpeedOfSoundAccess<T>()); 
    std::unique_ptr<ScalarAccessor<T>> mach = std::unique_ptr<ScalarAccessor<T>>(new MachNumberAccess<T>()); 
    write_scalar_field(f, fs, pressure, "pressure", "Float64", grid.num_cells());
    write_scalar_field(f, fs, temp, "temperature", "Float64", grid.num_cells());
    write_scalar_field(f, fs, density, "density", "Float64", grid.num_cells());
    write_scalar_field(f, fs, energy, "energy", "Float64", grid.num_cells());
    write_scalar_field(f, fs, a, "a", "Float64", grid.num_cells());
    write_scalar_field(f, fs, mach, "Mach", "Float64", grid.num_cells());
    write_vector_field(f, fs.vel, "velocity", "Float64", grid.num_cells());
    f << "</CellData>" << std::endl;
    f << "</Piece>" << std::endl;
    f << "</UnstructuredGrid>" << std::endl;
    f << "</VTKFile>";
    f.close();
    times_.push_back(time);
    dirs_.push_back(time_dir + "/block_0.vtu");
    return 0;
}

template <typename T>
void VtkOutput<T>::write_coordinating_file(std::string plot_dir) {
    std::ofstream plot_file(plot_dir + "/plot.pvd");
    plot_file << "<?xml version='1.0'?>" << std::endl;
    plot_file << "<VTKFile type='Collection' version='0.1' byte_order='LittleEndian'>" << std::endl;
    plot_file << "<Collection>" << std::endl;
    for (unsigned int i = 0; i < times_.size(); i++){
        plot_file << "<DataSet timestep='" << times_[i] << "' group='' part='0' file='" << dirs_[i] << "'/>" << std::endl;
    }
    plot_file << "</Collection>" << std::endl;
    plot_file << "</VTKFile>" << std::endl;
}

template class VtkOutput<double>;
