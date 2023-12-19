#include <io/accessor.h>

template <typename T>
T PressureAccess<T>::access(
    const FlowStates<T, array_layout, host_mem_space>& fs,
    const IdealGas<T>& gas_model, const int i) {
    (void)gas_model;
    return fs.gas.pressure(i);
}
template class PressureAccess<double>;

template <typename T>
T TemperatureAccess<T>::access(
    const FlowStates<T, array_layout, host_mem_space>& fs,
    const IdealGas<T>& gas_model, const int i) {
    (void)gas_model;
    return fs.gas.temp(i);
}
template class TemperatureAccess<double>;

template <typename T>
T DensityAccess<T>::access(
    const FlowStates<T, array_layout, host_mem_space>& fs,
    const IdealGas<T>& gas_model, const int i) {
    (void)gas_model;
    return fs.gas.rho(i);
}
template class DensityAccess<double>;

template <typename T>
T InternalEnergyAccess<T>::access(
    const FlowStates<T, array_layout, host_mem_space>& fs,
    const IdealGas<T>& gas_model, const int i) {
    (void)gas_model;
    return fs.gas.energy(i);
}
template class InternalEnergyAccess<double>;

template <typename T>
T SpeedOfSoundAccess<T>::access(
    const FlowStates<T, array_layout, host_mem_space>& fs,
    const IdealGas<T>& gas_model, const int i) {
    return gas_model.speed_of_sound(fs.gas, i);
}
template class SpeedOfSoundAccess<double>;

template <typename T>
T MachNumberAccess<T>::access(
    const FlowStates<T, array_layout, host_mem_space>& fs,
    const IdealGas<T>& gas_model, const int i) {
    T a = gas_model.speed_of_sound(fs.gas, i);
    T vx = fs.vel.x(i);
    T vy = fs.vel.y(i);
    T vz = fs.vel.z(i);
    T v_mag = Kokkos::sqrt(vx * vx + vy * vy + vz * vz);
    return v_mag / a;
}
template class MachNumberAccess<double>;

template <typename T>
std::map<std::string, std::shared_ptr<ScalarAccessor<T>>> get_accessors() {
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
get_accessors();
