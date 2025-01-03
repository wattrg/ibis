// #include <finite_volume/gradient.h>
#include <gas/transport_properties.h>
#include <io/accessor.h>

template <typename T>
T PressureAccess<T>::access(const FlowStates<T, array_layout, host_mem_space>& fs,
                            FiniteVolume<T>& fv,
                            const GridBlock<T, host_exec_space, array_layout>& grid,
                            const IdealGas<T>& gas_model, const int i) {
    (void)gas_model;
    (void)fv;
    (void)grid;
    return fs.gas.pressure(i);
}
template class PressureAccess<Ibis::real>;
template class PressureAccess<Ibis::dual>;

template <typename T>
T TemperatureAccess<T>::access(const FlowStates<T, array_layout, host_mem_space>& fs,
                               FiniteVolume<T>& fv,
                               const GridBlock<T, host_exec_space, array_layout>& grid,
                               const IdealGas<T>& gas_model, const int i) {
    (void)gas_model;
    (void)fv;
    (void)grid;
    return fs.gas.temp(i);
}
template class TemperatureAccess<Ibis::real>;
template class TemperatureAccess<Ibis::dual>;

template <typename T>
T DensityAccess<T>::access(const FlowStates<T, array_layout, host_mem_space>& fs,
                           FiniteVolume<T>& fv,
                           const GridBlock<T, host_exec_space, array_layout>& grid,
                           const IdealGas<T>& gas_model, const int i) {
    (void)gas_model;
    (void)fv;
    (void)grid;
    return fs.gas.rho(i);
}
template class DensityAccess<Ibis::real>;
template class DensityAccess<Ibis::dual>;

template <typename T>
T InternalEnergyAccess<T>::access(const FlowStates<T, array_layout, host_mem_space>& fs,
                                  FiniteVolume<T>& fv,
                                  const GridBlock<T, host_exec_space, array_layout>& grid,
                                  const IdealGas<T>& gas_model, const int i) {
    (void)gas_model;
    (void)fv;
    (void)grid;
    return fs.gas.energy(i);
}
template class InternalEnergyAccess<Ibis::real>;
template class InternalEnergyAccess<Ibis::dual>;

template <typename T>
T SpeedOfSoundAccess<T>::access(const FlowStates<T, array_layout, host_mem_space>& fs,
                                FiniteVolume<T>& fv,
                                const GridBlock<T, host_exec_space, array_layout>& grid,
                                const IdealGas<T>& gas_model, const int i) {
    (void)fv;
    (void)grid;
    return gas_model.speed_of_sound(fs.gas, i);
}
template class SpeedOfSoundAccess<Ibis::real>;
template class SpeedOfSoundAccess<Ibis::dual>;

template <typename T>
T MachNumberAccess<T>::access(const FlowStates<T, array_layout, host_mem_space>& fs,
                              FiniteVolume<T>& fv,
                              const GridBlock<T, host_exec_space, array_layout>& grid,
                              const IdealGas<T>& gas_model, const int i) {
    (void)fv;
    (void)grid;
    T a = gas_model.speed_of_sound(fs.gas, i);
    T vx = fs.vel.x(i);
    T vy = fs.vel.y(i);
    T vz = fs.vel.z(i);
    T v_mag = Ibis::sqrt(vx * vx + vy * vy + vz * vz);
    return v_mag / a;
}
template class MachNumberAccess<Ibis::real>;
template class MachNumberAccess<Ibis::dual>;

template <typename T>
T VolumeAccess<T>::access(const FlowStates<T, array_layout, host_mem_space>& fs,
                          FiniteVolume<T>& fv,
                          const GridBlock<T, host_exec_space, array_layout>& grid,
                          const IdealGas<T>& gas_model, const int i) {
    (void)fv;
    (void)fs;
    (void)gas_model;
    return grid.cells().volume(i);
}
template class VolumeAccess<Ibis::real>;
template class VolumeAccess<Ibis::dual>;

template <typename T>
Vector3<T> VelocityAccess<T>::access(
    const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
    const GridBlock<T, host_exec_space, array_layout>& grid, const IdealGas<T>& gas_model,
    const int i) {
    (void)gas_model;
    (void)grid;
    (void)fv;
    T x = fs.vel.x(i);
    T y = fs.vel.y(i);
    T z = fs.vel.z(i);
    return Vector3<T>(x, y, z);
}
template class VelocityAccess<Ibis::real>;
template class VelocityAccess<Ibis::dual>;

template <typename T>
void ViscousGradVxAccess<T>::init(const FlowStates<T, array_layout, host_mem_space>& fs,
                                  FiniteVolume<T>& fv, const GridBlock<T>& grid,
                                  const IdealGas<T>& gas_model,
                                  const TransportProperties<T>& trans_prop) {
    (void)gas_model;
    (void)grid;
    FlowStates<T> fs_dev = FlowStates<T>(fs.number_flow_states());
    fs_dev.deep_copy(fs);
    fv.compute_viscous_gradient(fs_dev, grid, gas_model, trans_prop);
    grad_vx_ = fv.cell_gradients().vx.host_mirror();
    grad_vx_.deep_copy(fv.cell_gradients().vx);
}

template <typename T>
Vector3<T> ViscousGradVxAccess<T>::access(
    const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
    const GridBlock<T, host_exec_space, array_layout>& grid, const IdealGas<T>& gas_model,
    const int i) {
    (void)gas_model;
    (void)grid;
    (void)fs;
    (void)fv;
    T grad_x = grad_vx_.x(i);
    T grad_y = grad_vx_.y(i);
    T grad_z = grad_vx_.z(i);
    return Vector3<T>(grad_x, grad_y, grad_z);
}
template class ViscousGradVxAccess<Ibis::real>;
template class ViscousGradVxAccess<Ibis::dual>;

template <typename T>
void ViscousGradVyAccess<T>::init(const FlowStates<T, array_layout, host_mem_space>& fs,
                                  FiniteVolume<T>& fv, const GridBlock<T>& grid,
                                  const IdealGas<T>& gas_model,
                                  const TransportProperties<T>& trans_prop) {
    (void)gas_model;
    (void)grid;
    FlowStates<T> fs_dev = FlowStates<T>(fs.number_flow_states());
    fs_dev.deep_copy(fs);
    fv.compute_viscous_gradient(fs_dev, grid, gas_model, trans_prop);
    grad_vy_ = fv.cell_gradients().vy.host_mirror();
    grad_vy_.deep_copy(fv.cell_gradients().vy);
}

template <typename T>
Vector3<T> ViscousGradVyAccess<T>::access(
    const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
    const GridBlock<T, host_exec_space, array_layout>& grid, const IdealGas<T>& gas_model,
    const int i) {
    (void)gas_model;
    (void)fs;
    (void)fv;
    (void)grid;
    T grad_x = grad_vy_.x(i);
    T grad_y = grad_vy_.y(i);
    T grad_z = grad_vy_.z(i);
    return Vector3<T>(grad_x, grad_y, grad_z);
}
template class ViscousGradVyAccess<Ibis::real>;
template class ViscousGradVyAccess<Ibis::dual>;

template <typename T>
void ViscousGradVzAccess<T>::init(const FlowStates<T, array_layout, host_mem_space>& fs,
                                  FiniteVolume<T>& fv, const GridBlock<T>& grid,
                                  const IdealGas<T>& gas_model,
                                  const TransportProperties<T>& trans_prop) {
    (void)gas_model;
    (void)grid;
    FlowStates<T> fs_dev = FlowStates<T>(fs.number_flow_states());
    fs_dev.deep_copy(fs);
    fv.compute_viscous_gradient(fs_dev, grid, gas_model, trans_prop);
    grad_vz_ = fv.cell_gradients().vz.host_mirror();
    grad_vz_.deep_copy(fv.cell_gradients().vz);
}

template <typename T>
Vector3<T> ViscousGradVzAccess<T>::access(
    const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
    const GridBlock<T, host_exec_space, array_layout>& grid, const IdealGas<T>& gas_model,
    const int i) {
    (void)gas_model;
    (void)fs;
    (void)grid;
    (void)fv;
    T grad_x = grad_vz_.x(i);
    T grad_y = grad_vz_.y(i);
    T grad_z = grad_vz_.z(i);
    return Vector3<T>(grad_x, grad_y, grad_z);
}
template class ViscousGradVzAccess<Ibis::real>;
template class ViscousGradVzAccess<Ibis::dual>;

template <typename T>
void ConvectiveGradVxAccess<T>::init(
    const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
    const GridBlock<T>& grid, const IdealGas<T>& gas_model,
    const TransportProperties<T>& trans_prop) {
    (void)gas_model;
    (void)grid;
    FlowStates<T> fs_dev = FlowStates<T>(fs.number_flow_states());
    fs_dev.deep_copy(fs);
    fv.compute_convective_gradient(fs_dev, grid, gas_model, trans_prop);
    grad_vx_ = fv.cell_gradients().vx.host_mirror();
    grad_vx_.deep_copy(fv.cell_gradients().vx);
}

template <typename T>
Vector3<T> ConvectiveGradVxAccess<T>::access(
    const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
    const GridBlock<T, host_exec_space, array_layout>& grid, const IdealGas<T>& gas_model,
    const int i) {
    (void)gas_model;
    (void)fs;
    (void)fv;
    (void)grid;
    T grad_x = grad_vx_.x(i);
    T grad_y = grad_vx_.y(i);
    T grad_z = grad_vx_.z(i);
    return Vector3<T>(grad_x, grad_y, grad_z);
}
template class ConvectiveGradVxAccess<Ibis::real>;
template class ConvectiveGradVxAccess<Ibis::dual>;

template <typename T>
void ConvectiveGradVyAccess<T>::init(
    const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
    const GridBlock<T>& grid, const IdealGas<T>& gas_model,
    const TransportProperties<T>& trans_prop) {
    (void)gas_model;
    (void)grid;
    FlowStates<T> fs_dev = FlowStates<T>(fs.number_flow_states());
    fs_dev.deep_copy(fs);
    fv.compute_convective_gradient(fs_dev, grid, gas_model, trans_prop);
    grad_vy_ = fv.cell_gradients().vy.host_mirror();
    grad_vy_.deep_copy(fv.cell_gradients().vy);
}

template <typename T>
Vector3<T> ConvectiveGradVyAccess<T>::access(
    const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
    const GridBlock<T, host_exec_space, array_layout>& grid, const IdealGas<T>& gas_model,
    const int i) {
    (void)gas_model;
    (void)grid;
    (void)fs;
    (void)fv;
    T grad_x = grad_vy_.x(i);
    T grad_y = grad_vy_.y(i);
    T grad_z = grad_vy_.z(i);
    return Vector3<T>(grad_x, grad_y, grad_z);
}
template class ConvectiveGradVyAccess<Ibis::real>;
template class ConvectiveGradVyAccess<Ibis::dual>;

template <typename T>
void ConvectiveGradVzAccess<T>::init(
    const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
    const GridBlock<T>& grid, const IdealGas<T>& gas_model,
    const TransportProperties<T>& trans_prop) {
    (void)gas_model;
    (void)grid;
    FlowStates<T> fs_dev = FlowStates<T>(fs.number_flow_states());
    fs_dev.deep_copy(fs);
    fv.compute_convective_gradient(fs_dev, grid, gas_model, trans_prop);
    grad_vz_ = fv.cell_gradients().vz.host_mirror();
    grad_vz_.deep_copy(fv.cell_gradients().vz);
}

template <typename T>
Vector3<T> ConvectiveGradVzAccess<T>::access(
    const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
    const GridBlock<T, host_exec_space, array_layout>& grid, const IdealGas<T>& gas_model,
    const int i) {
    (void)grid;
    (void)grid;
    (void)gas_model;
    (void)fs;
    (void)fv;
    T grad_x = grad_vz_.x(i);
    T grad_y = grad_vz_.y(i);
    T grad_z = grad_vz_.z(i);
    return Vector3<T>(grad_x, grad_y, grad_z);
}
template class ConvectiveGradVzAccess<Ibis::real>;
template class ConvectiveGradVzAccess<Ibis::dual>;

template <typename T>
Vector3<T> CellCentreAccess<T>::access(
    const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
    const GridBlock<T, host_exec_space, array_layout>& grid, const IdealGas<T>& gas_model,
    const int i) {
    (void)gas_model;
    (void)grid;
    (void)fs;
    (void)fv;
    T x = grid.cells().centroids().x(i);
    T y = grid.cells().centroids().y(i);
    T z = grid.cells().centroids().z(i);
    return Vector3<T>(x, y, z);
}
template class CellCentreAccess<Ibis::real>;
template class CellCentreAccess<Ibis::dual>;

template <typename T>
std::map<std::string, std::shared_ptr<ScalarAccessor<T>>> get_scalar_accessors() {
    return {
        {"pressure", std::shared_ptr<ScalarAccessor<T>>(new PressureAccess<T>())},
        {"temperature", std::shared_ptr<ScalarAccessor<T>>(new TemperatureAccess<T>())},
        {"density", std::shared_ptr<ScalarAccessor<T>>(new DensityAccess<T>())},
        {"energy", std::shared_ptr<ScalarAccessor<T>>(new InternalEnergyAccess<T>())},
        {"a", std::shared_ptr<ScalarAccessor<T>>(new SpeedOfSoundAccess<T>())},
        {"Mach", std::shared_ptr<ScalarAccessor<T>>(new MachNumberAccess<T>())}};
}
template std::map<std::string, std::shared_ptr<ScalarAccessor<Ibis::real>>>
get_scalar_accessors();
template std::map<std::string, std::shared_ptr<ScalarAccessor<Ibis::dual>>>
get_scalar_accessors();

template <typename T>
std::map<std::string, std::shared_ptr<VectorAccessor<T>>> get_vector_accessors() {
    return {{"velocity", std::shared_ptr<VectorAccessor<T>>(new VelocityAccess<T>())}};
}
template std::map<std::string, std::shared_ptr<VectorAccessor<Ibis::real>>>
get_vector_accessors();
template std::map<std::string, std::shared_ptr<VectorAccessor<Ibis::dual>>>
get_vector_accessors();
