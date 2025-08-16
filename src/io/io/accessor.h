#ifndef ACCESSOR_H
#define ACCESSOR_H

#include <finite_volume/finite_volume.h>
#include <gas/flow_state.h>
#include <gas/gas_model.h>
#include <grid/gradient.h>
#include <grid/grid.h>
#include <util/vector3.h>

#include <Kokkos_Core.hpp>
#include <map>

#include "gas/transport_properties.h"

using array_layout = Kokkos::DefaultExecutionSpace::array_layout;
using host_mem_space = Kokkos::DefaultHostExecutionSpace::memory_space;
using host_exec_space = Kokkos::DefaultHostExecutionSpace;

template <typename T, class MemModel>
class ScalarAccessor {
public:
    virtual ~ScalarAccessor(){};

    virtual void init(const FlowStates<T, array_layout, host_mem_space>& fs,
                      FiniteVolume<T, MemModel>& fv, const GridBlock<MemModel, T>& grid,
                      const IdealGas<T>& gas_model,
                      const TransportProperties<T>& trans_prop) {
        (void)grid;
        (void)fs;
        (void)fv;
        (void)gas_model;
        (void)trans_prop;
    }

    virtual T access(const FlowStates<T, array_layout, host_mem_space>& fs,
                     FiniteVolume<T, MemModel>& fv,
                     const GridBlock<MemModel, T, host_exec_space, array_layout>& grid,
                     const IdealGas<T>& gas_model, const int i) = 0;
};

template <typename T, class MemModel>
class VectorAccessor {
public:
    virtual ~VectorAccessor(){};

    virtual void init(const FlowStates<T, array_layout, host_mem_space>& fs,
                      FiniteVolume<T, MemModel>& fv, const GridBlock<MemModel, T>& grid,
                      const IdealGas<T>& gas_model,
                      const TransportProperties<T>& trans_prop) {
        (void)grid;
        (void)fs;
        (void)fv;
        (void)gas_model;
        (void)trans_prop;
    }

    virtual Vector3<T> access(const FlowStates<T, array_layout, host_mem_space>& fs,
                              FiniteVolume<T, MemModel>& fv,
                              const GridBlock<MemModel, T, host_exec_space, array_layout>& grid,
                              const IdealGas<T>& gas_model, const int i) = 0;
};

template <typename T, class MemModel>
class PressureAccess : public ScalarAccessor<T, MemModel> {
    T access(const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T, MemModel>& fv,
             const GridBlock<MemModel, T, host_exec_space, array_layout>& grid,
             const IdealGas<T>& gas_model, const int i);
};

template <typename T, class MemModel>
class TemperatureAccess : public ScalarAccessor<T, MemModel> {
    T access(const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T, MemModel>& fv,
             const GridBlock<MemModel, T, host_exec_space, array_layout>& grid,
             const IdealGas<T>& gas_model, const int i);
};

template <typename T, class MemModel>
class DensityAccess : public ScalarAccessor<T, MemModel> {
    T access(const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T, MemModel>& fv,
             const GridBlock<MemModel, T, host_exec_space, array_layout>& grid,
             const IdealGas<T>& gas_model, const int i);
};

template <typename T, class MemModel>
class InternalEnergyAccess : public ScalarAccessor<T, MemModel> {
    T access(const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T, MemModel>& fv,
             const GridBlock<MemModel, T, host_exec_space, array_layout>& grid,
             const IdealGas<T>& gas_model, const int i);
};

template <typename T, class MemModel>
class SpeedOfSoundAccess : public ScalarAccessor<T, MemModel> {
    T access(const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T, MemModel>& fv,
             const GridBlock<MemModel, T, host_exec_space, array_layout>& grid,
             const IdealGas<T>& gas_model, const int i);
};

template <typename T, class MemModel>
class MachNumberAccess : public ScalarAccessor<T, MemModel> {
    T access(const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T, MemModel>& fv,
             const GridBlock<MemModel, T, host_exec_space, array_layout>& grid,
             const IdealGas<T>& gas_model, const int i);
};

template <typename T, class MemModel>
class VolumeAccess : public ScalarAccessor<T, MemModel> {
    T access(const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T, MemModel>& fv,
             const GridBlock<MemModel, T, host_exec_space, array_layout>& grid,
             const IdealGas<T>& gas_model, const int i);
};

template <typename T, class MemModel>
class VelocityAccess : public VectorAccessor<T, MemModel> {
    Vector3<T> access(const FlowStates<T, array_layout, host_mem_space>& fs,
                      FiniteVolume<T, MemModel>& fv,
                      const GridBlock<MemModel, T, host_exec_space, array_layout>& grid,
                      const IdealGas<T>& gas_model, const int i);
};

// viscous gradients
template <typename T, class MemModel>
class ViscousGradVxAccess : public VectorAccessor<T, MemModel> {
public:
    void init(const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T, MemModel>& fv,
              const GridBlock<MemModel, T>& grid, const IdealGas<T>& gas_model,
              const TransportProperties<T>& trans_prop) override;

    Vector3<T> access(const FlowStates<T, array_layout, host_mem_space>& fs,
                      FiniteVolume<T, MemModel>& fv,
                      const GridBlock<MemModel, T, host_exec_space, array_layout>& grid,
                      const IdealGas<T>& gas_model, const int i) override;

private:
    Vector3s<T, array_layout, host_mem_space> grad_vx_;
};

template <typename T, class MemModel>
class ViscousGradVyAccess : public VectorAccessor<T, MemModel> {
public:
    void init(const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T, MemModel>& fv,
              const GridBlock<MemModel, T>& grid, const IdealGas<T>& gas_model,
              const TransportProperties<T>& trans_prop) override;

    Vector3<T> access(const FlowStates<T, array_layout, host_mem_space>& fs,
                      FiniteVolume<T, MemModel>& fv,
                      const GridBlock<MemModel, T, host_exec_space, array_layout>& grid,
                      const IdealGas<T>& gas_model, const int i) override;

private:
    Vector3s<T, array_layout, host_mem_space> grad_vy_;
};

template <typename T, class MemModel>
class ViscousGradVzAccess : public VectorAccessor<T, MemModel> {
public:
    void init(const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T, MemModel>& fv,
              const GridBlock<MemModel, T>& grid, const IdealGas<T>& gas_model,
              const TransportProperties<T>& trans_prop) override;

    Vector3<T> access(const FlowStates<T, array_layout, host_mem_space>& fs,
                      FiniteVolume<T, MemModel>& fv,
                      const GridBlock<MemModel, T, host_exec_space, array_layout>& grid,
                      const IdealGas<T>& gas_model, const int i) override;

private:
    Vector3s<T, array_layout, host_mem_space> grad_vz_;
};

// convective gradients
template <typename T, class MemModel>
class ConvectiveGradVxAccess : public VectorAccessor<T, MemModel> {
public:
    void init(const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T, MemModel>& fv,
              const GridBlock<MemModel, T>& grid, const IdealGas<T>& gas_model,
              const TransportProperties<T>& trans_prop) override;

    Vector3<T> access(const FlowStates<T, array_layout, host_mem_space>& fs,
                      FiniteVolume<T, MemModel>& fv,
                      const GridBlock<MemModel, T, host_exec_space, array_layout>& grid,
                      const IdealGas<T>& gas_model, const int i) override;

private:
    Vector3s<T, array_layout, host_mem_space> grad_vx_;
};

template <typename T, class MemModel>
class ConvectiveGradVyAccess : public VectorAccessor<T, MemModel> {
public:
    void init(const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T, MemModel>& fv,
              const GridBlock<MemModel, T>& grid, const IdealGas<T>& gas_model,
              const TransportProperties<T>& trans_prop) override;

    Vector3<T> access(const FlowStates<T, array_layout, host_mem_space>& fs,
                      FiniteVolume<T, MemModel>& fv,
                      const GridBlock<MemModel, T, host_exec_space, array_layout>& grid,
                      const IdealGas<T>& gas_model, const int i) override;

private:
    Vector3s<T, array_layout, host_mem_space> grad_vy_;
};

template <typename T, class MemModel>
class ConvectiveGradVzAccess : public VectorAccessor<T, MemModel> {
public:
    void init(const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T, MemModel>& fv,
              const GridBlock<MemModel, T>& grid, const IdealGas<T>& gas_model,
              const TransportProperties<T>& trans_prop) override;

    Vector3<T> access(const FlowStates<T, array_layout, host_mem_space>& fs,
                      FiniteVolume<T, MemModel>& fv,
                      const GridBlock<MemModel, T, host_exec_space, array_layout>& grid,
                      const IdealGas<T>& gas_model, const int i) override;

private:
    Vector3s<T, array_layout, host_mem_space> grad_vz_;
};

template <typename T, class MemModel>
class CellCentreAccess : public VectorAccessor<T, MemModel> {
public:
    Vector3<T> access(const FlowStates<T, array_layout, host_mem_space>& fs,
                      FiniteVolume<T, MemModel>& fv,
                      const GridBlock<MemModel, T, host_exec_space, array_layout>& grid,
                      const IdealGas<T>& gas_model, const int i) override;
};

template <typename T, class MemModel>
std::map<std::string, std::shared_ptr<ScalarAccessor<T, MemModel>>> get_scalar_accessors();

template <typename T, class MemModel>
std::map<std::string, std::shared_ptr<VectorAccessor<T, MemModel>>> get_vector_accessors();

#endif
