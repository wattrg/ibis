#include <io/accessor.h>

#include "finite_volume/gradient.h"

template <typename T>
T PressureAccess<T>::access(
    const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
    const IdealGas<T>& gas_model, const int i) {
    (void)gas_model;
    (void)fv;
    return fs.gas.pressure(i);
}
template class PressureAccess<double>;

template <typename T>
T TemperatureAccess<T>::access(
    const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
    const IdealGas<T>& gas_model, const int i) {
    (void)gas_model;
    (void)fv;
    return fs.gas.temp(i);
}
template class TemperatureAccess<double>;

template <typename T>
T DensityAccess<T>::access(
    const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
    const IdealGas<T>& gas_model, const int i) {
    (void)gas_model;
    (void)fv;
    return fs.gas.rho(i);
}
template class DensityAccess<double>;

template <typename T>
T InternalEnergyAccess<T>::access(
    const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
    const IdealGas<T>& gas_model, const int i) {
    (void)gas_model;
    (void)fv;
    return fs.gas.energy(i);
}
template class InternalEnergyAccess<double>;

template <typename T>
T SpeedOfSoundAccess<T>::access(
    const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
    const IdealGas<T>& gas_model, const int i) {
    (void)fv;
    return gas_model.speed_of_sound(fs.gas, i);
}
template class SpeedOfSoundAccess<double>;

template <typename T>
T MachNumberAccess<T>::access(
    const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
    const IdealGas<T>& gas_model, const int i) {
    (void)fv;
    T a = gas_model.speed_of_sound(fs.gas, i);
    T vx = fs.vel.x(i);
    T vy = fs.vel.y(i);
    T vz = fs.vel.z(i);
    T v_mag = Kokkos::sqrt(vx * vx + vy * vy + vz * vz);
    return v_mag / a;
}
template class MachNumberAccess<double>;

template <typename T>
Vector3<T> VelocityAccess<T>::access(
    const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
    const IdealGas<T>& gas_model, const int i) {
    (void)gas_model;
    (void)fv;
    T x = fs.vel.x(i);
    T y = fs.vel.y(i);
    T z = fs.vel.z(i);
    return Vector3<T>(x, y, z);
}
template class VelocityAccess<double>;

template <typename T>
void GradVxAccess<T>::init(
    const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
    const GridBlock<T>& grid, const IdealGas<T>& gas_model) {
    (void)gas_model;
    FlowStates<T> fs_dev = FlowStates<T>(fs.number_flow_states());
    fs_dev.deep_copy(fs);
    fv.compute_viscous_gradient(fs_dev, grid);
    grad_vx_ = fv.cell_gradients().vx.host_mirror();
    grad_vx_.deep_copy(fv.cell_gradients().vx);
}

template <typename T>
Vector3<T> GradVxAccess<T>::access(
    const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
    const IdealGas<T>& gas_model, const int i) {
    (void)gas_model;
    (void)fs;
    (void)fv;
    T grad_x = grad_vx_.x(i);
    T grad_y = grad_vx_.y(i);
    T grad_z = grad_vx_.z(i);
    return Vector3<T>(grad_x, grad_y, grad_z);
}
template class GradVxAccess<double>;

template <typename T>
void GradVyAccess<T>::init(
    const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
    const GridBlock<T>& grid, const IdealGas<T>& gas_model) {
    (void)gas_model;
    FlowStates<T> fs_dev = FlowStates<T>(fs.number_flow_states());
    fs_dev.deep_copy(fs);
    fv.compute_viscous_gradient(fs_dev, grid);
    grad_vy_ = fv.cell_gradients().vy.host_mirror();
    grad_vy_.deep_copy(fv.cell_gradients().vy);
}

template <typename T>
Vector3<T> GradVyAccess<T>::access(
    const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
    const IdealGas<T>& gas_model, const int i) {
    (void)gas_model;
    (void)fs;
    (void)fv;
    T grad_x = grad_vy_.x(i);
    T grad_y = grad_vy_.y(i);
    T grad_z = grad_vy_.z(i);
    return Vector3<T>(grad_x, grad_y, grad_z);
}
template class GradVyAccess<double>;

template <typename T>
void GradVzAccess<T>::init(
    const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
    const GridBlock<T>& grid, const IdealGas<T>& gas_model) {
    (void)gas_model;
    FlowStates<T> fs_dev = FlowStates<T>(fs.number_flow_states());
    fs_dev.deep_copy(fs);
    fv.compute_viscous_gradient(fs_dev, grid);
    grad_vz_ = fv.cell_gradients().vz.host_mirror();
    grad_vz_.deep_copy(fv.cell_gradients().vz);
}

template <typename T>
Vector3<T> GradVzAccess<T>::access(
    const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
    const IdealGas<T>& gas_model, const int i) {
    (void)gas_model;
    (void)fs;
    (void)fv;
    T grad_x = grad_vz_.x(i);
    T grad_y = grad_vz_.y(i);
    T grad_z = grad_vz_.z(i);
    return Vector3<T>(grad_x, grad_y, grad_z);
}
template class GradVzAccess<double>;

template <typename T>
std::map<std::string, std::shared_ptr<ScalarAccessor<T>>>
get_scalar_accessors() {
    return {
        {"pressure",
         std::shared_ptr<ScalarAccessor<T>>(new PressureAccess<T>())},
        {"temperature",
         std::shared_ptr<ScalarAccessor<T>>(new TemperatureAccess<T>())},
        {"density", std::shared_ptr<ScalarAccessor<T>>(new DensityAccess<T>())},
        {"energy",
         std::shared_ptr<ScalarAccessor<T>>(new InternalEnergyAccess<T>())},
        {"a", std::shared_ptr<ScalarAccessor<T>>(new SpeedOfSoundAccess<T>())},
        {"Mach",
         std::shared_ptr<ScalarAccessor<T>>(new MachNumberAccess<T>())}};
}
template std::map<std::string, std::shared_ptr<ScalarAccessor<double>>>
get_scalar_accessors();

template <typename T>
std::map<std::string, std::shared_ptr<VectorAccessor<T>>>
get_vector_accessors() {
    return {{"velocity",
             std::shared_ptr<VectorAccessor<T>>(new VelocityAccess<T>())}};
}
template std::map<std::string, std::shared_ptr<VectorAccessor<double>>>
get_vector_accessors();
