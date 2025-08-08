// #include <finite_volume/gradient.h>
#include <gas/transport_properties.h>
#include <io/accessor.h>

template <typename T, class MemModel>
T PressureAccess<T, MemModel>::access(const FlowStates<T, array_layout, host_mem_space>& fs,
                            FiniteVolume<T>& fv,
                            const GridBlock<MemModel, T, host_exec_space, array_layout>& grid,
                            const IdealGas<T>& gas_model, const int i) {
    (void)gas_model;
    (void)fv;
    (void)grid;
    return fs.gas.pressure(i);
}
template class PressureAccess<Ibis::real, SharedMem>;
template class PressureAccess<Ibis::dual, SharedMem>;

template <typename T, class MemModel>
T TemperatureAccess<T, MemModel>::access(const FlowStates<T, array_layout, host_mem_space>& fs,
                               FiniteVolume<T>& fv,
                               const GridBlock<MemModel, T, host_exec_space, array_layout>& grid,
                               const IdealGas<T>& gas_model, const int i) {
    (void)gas_model;
    (void)fv;
    (void)grid;
    return fs.gas.temp(i);
}
template class TemperatureAccess<Ibis::real, SharedMem>;
template class TemperatureAccess<Ibis::dual, SharedMem>;

template <typename T, class MemModel>
T DensityAccess<T, MemModel>::access(const FlowStates<T, array_layout, host_mem_space>& fs,
                           FiniteVolume<T>& fv,
                           const GridBlock<MemModel, T, host_exec_space, array_layout>& grid,
                           const IdealGas<T>& gas_model, const int i) {
    (void)gas_model;
    (void)fv;
    (void)grid;
    return fs.gas.rho(i);
}
template class DensityAccess<Ibis::real, SharedMem>;
template class DensityAccess<Ibis::dual, SharedMem>;

template <typename T, class MemModel>
T InternalEnergyAccess<T, MemModel>::access(const FlowStates<T, array_layout, host_mem_space>& fs,
                                  FiniteVolume<T>& fv,
                                  const GridBlock<MemModel, T, host_exec_space, array_layout>& grid,
                                  const IdealGas<T>& gas_model, const int i) {
    (void)gas_model;
    (void)fv;
    (void)grid;
    return fs.gas.energy(i);
}
template class InternalEnergyAccess<Ibis::real, SharedMem>;
template class InternalEnergyAccess<Ibis::dual, SharedMem>;

template <typename T, class MemModel>
T SpeedOfSoundAccess<T, MemModel>::access(const FlowStates<T, array_layout, host_mem_space>& fs,
                                FiniteVolume<T>& fv,
                                const GridBlock<MemModel, T, host_exec_space, array_layout>& grid,
                                const IdealGas<T>& gas_model, const int i) {
    (void)fv;
    (void)grid;
    return gas_model.speed_of_sound(fs.gas, i);
}
template class SpeedOfSoundAccess<Ibis::real, SharedMem>;
template class SpeedOfSoundAccess<Ibis::dual, SharedMem>;

template <typename T, class MemModel>
T MachNumberAccess<T, MemModel>::access(const FlowStates<T, array_layout, host_mem_space>& fs,
                              FiniteVolume<T>& fv,
                              const GridBlock<MemModel, T, host_exec_space, array_layout>& grid,
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
template class MachNumberAccess<Ibis::real, SharedMem>;
template class MachNumberAccess<Ibis::dual, SharedMem>;

template <typename T, class MemModel>
T VolumeAccess<T, MemModel>::access(const FlowStates<T, array_layout, host_mem_space>& fs,
                          FiniteVolume<T>& fv,
                          const GridBlock<MemModel, T, host_exec_space, array_layout>& grid,
                          const IdealGas<T>& gas_model, const int i) {
    (void)fv;
    (void)fs;
    (void)gas_model;
    return grid.cells().volume(i);
}
template class VolumeAccess<Ibis::real, SharedMem>;
template class VolumeAccess<Ibis::dual, SharedMem>;

template <typename T, class MemModel>
Vector3<T> VelocityAccess<T, MemModel>::access(
    const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
    const GridBlock<MemModel, T, host_exec_space, array_layout>& grid, const IdealGas<T>& gas_model,
    const int i) {
    (void)gas_model;
    (void)grid;
    (void)fv;
    T x = fs.vel.x(i);
    T y = fs.vel.y(i);
    T z = fs.vel.z(i);
    return Vector3<T>(x, y, z);
}
template class VelocityAccess<Ibis::real, SharedMem>;
template class VelocityAccess<Ibis::dual, SharedMem>;

template <typename T, class MemModel>
void ViscousGradVxAccess<T, MemModel>::init(const FlowStates<T, array_layout, host_mem_space>& fs,
                                  FiniteVolume<T>& fv, const GridBlock<MemModle, T>& grid,
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

template <typename T, class MemModel>
Vector3<T> ViscousGradVxAccess<T, MemModel>::access(
    const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
    const GridBlock<MemModel, T, host_exec_space, array_layout>& grid, const IdealGas<T>& gas_model,
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
template class ViscousGradVxAccess<Ibis::real, SharedMem>;
template class ViscousGradVxAccess<Ibis::dual, SharedMem>;

template <typename T, class MemModel>
void ViscousGradVyAccess<T, MemModel>::init(const FlowStates<T, array_layout, host_mem_space>& fs,
                                  FiniteVolume<T>& fv, const GridBlock<MemModel, T>& grid,
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

template <typename T, class MemModel>
Vector3<T> ViscousGradVyAccess<T, MemModel>::access(
    const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
    const GridBlock<MemModel, T, host_exec_space, array_layout>& grid, const IdealGas<T>& gas_model,
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
template class ViscousGradVyAccess<Ibis::real, SharedMem>;
template class ViscousGradVyAccess<Ibis::dual, SharedMem>;

template <typename T, class MemModel>
void ViscousGradVzAccess<T, MemModel>::init(const FlowStates<T, array_layout, host_mem_space>& fs,
                                  FiniteVolume<T>& fv, const GridBlock<MemModel, T>& grid,
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

template <typename T, class MemModel>
Vector3<T> ViscousGradVzAccess<T, MemModel>::access(
    const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
    const GridBlock<MemModel, T, host_exec_space, array_layout>& grid, const IdealGas<T>& gas_model,
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
template class ViscousGradVzAccess<Ibis::real, SharedMem>;
template class ViscousGradVzAccess<Ibis::dual, SharedMem>;

template <typename T, class MemModel>
void ConvectiveGradVxAccess<T, MemModel>::init(
    const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
    const GridBlock<MemModel, T>& grid, const IdealGas<T>& gas_model,
    const TransportProperties<T>& trans_prop) {
    (void)gas_model;
    (void)grid;
    FlowStates<T> fs_dev = FlowStates<T>(fs.number_flow_states());
    fs_dev.deep_copy(fs);
    fv.compute_convective_gradient(fs_dev, grid, gas_model, trans_prop);
    grad_vx_ = fv.cell_gradients().vx.host_mirror();
    grad_vx_.deep_copy(fv.cell_gradients().vx);
}

template <typename T, class MemModel>
Vector3<T> ConvectiveGradVxAccess<T, MemModel>::access(
    const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
    const GridBlock<MemModel, T, host_exec_space, array_layout>& grid, const IdealGas<T>& gas_model,
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
template class ConvectiveGradVxAccess<Ibis::real, SharedMem>;
template class ConvectiveGradVxAccess<Ibis::dual, SharedMem>;

template <typename T, class MemModel>
void ConvectiveGradVyAccess<T, MemModel>::init(
    const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
    const GridBlock<MemModel, T>& grid, const IdealGas<T>& gas_model,
    const TransportProperties<T>& trans_prop) {
    (void)gas_model;
    (void)grid;
    FlowStates<T> fs_dev = FlowStates<T>(fs.number_flow_states());
    fs_dev.deep_copy(fs);
    fv.compute_convective_gradient(fs_dev, grid, gas_model, trans_prop);
    grad_vy_ = fv.cell_gradients().vy.host_mirror();
    grad_vy_.deep_copy(fv.cell_gradients().vy);
}

template <typename T, class MemModel>
Vector3<T> ConvectiveGradVyAccess<T, MemModel>::access(
    const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
    const GridBlock<MemModel, T, host_exec_space, array_layout>& grid, const IdealGas<T>& gas_model,
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
template class ConvectiveGradVyAccess<Ibis::real, SharedMem>;
template class ConvectiveGradVyAccess<Ibis::dual, SharedMem>;

template <typename T, class MemModel>
void ConvectiveGradVzAccess<T, MemModel>::init(
    const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
    const GridBlock<MemModel, T>& grid, const IdealGas<T>& gas_model,
    const TransportProperties<T>& trans_prop) {
    (void)gas_model;
    (void)grid;
    FlowStates<T> fs_dev = FlowStates<T>(fs.number_flow_states());
    fs_dev.deep_copy(fs);
    fv.compute_convective_gradient(fs_dev, grid, gas_model, trans_prop);
    grad_vz_ = fv.cell_gradients().vz.host_mirror();
    grad_vz_.deep_copy(fv.cell_gradients().vz);
}

template <typename T, class MemModel>
Vector3<T> ConvectiveGradVzAccess<T, MemModel>::access(
    const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
    const GridBlock<MemModel, T, host_exec_space, array_layout>& grid, const IdealGas<T>& gas_model,
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
template class ConvectiveGradVzAccess<Ibis::real, SharedMem>;
template class ConvectiveGradVzAccess<Ibis::dual, SharedMem>;

template <typename T, class MemModel>
Vector3<T> CellCentreAccess<T, MemModel>::access(
    const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
    const GridBlock<MemModel, T, host_exec_space, array_layout>& grid, const IdealGas<T>& gas_model,
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
template class CellCentreAccess<Ibis::real, SharedMem>;
template class CellCentreAccess<Ibis::dual, SharedMem>;

template <typename T, class MemModel>
std::map<std::string, std::shared_ptr<ScalarAccessor<T, MemModel>>> get_scalar_accessors() {
    return {
        {"pressure", std::shared_ptr<ScalarAccessor<T, MemModel>>(new PressureAccess<T, MemModel>())},
        {"temperature", std::shared_ptr<ScalarAccessor<T, MemModel>>(new TemperatureAccess<T, MemModel>())},
        {"density", std::shared_ptr<ScalarAccessor<T, MemModel>>(new DensityAccess<T, MemModel>())},
        {"energy", std::shared_ptr<ScalarAccessor<T, MemModel>>(new InternalEnergyAccess<T, MemModel>())},
        {"a", std::shared_ptr<ScalarAccessor<T, MemModel>>(new SpeedOfSoundAccess<T, MemModel>())},
        {"Mach", std::shared_ptr<ScalarAccessor<T, MemModel>>(new MachNumberAccess<T, MemModel>())}};
}
template std::map<std::string, std::shared_ptr<ScalarAccessor<Ibis::real, SharedMem>>>
get_scalar_accessors();
template std::map<std::string, std::shared_ptr<ScalarAccessor<Ibis::dual, SharedMem>>>
get_scalar_accessors();

template <typename T, class MemModel>
std::map<std::string, std::shared_ptr<VectorAccessor<T, MemModel>>> get_vector_accessors() {
    return {{"velocity", std::shared_ptr<VectorAccessor<T>>(new VelocityAccess<T, MemModel>())}};
}
template std::map<std::string, std::shared_ptr<VectorAccessor<Ibis::real, SharedMem>>>
get_vector_accessors();
template std::map<std::string, std::shared_ptr<VectorAccessor<Ibis::dual, SharedMem>>>
get_vector_accessors();
