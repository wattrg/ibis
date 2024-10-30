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

template <typename T>
class ScalarAccessor {
public:
    virtual ~ScalarAccessor(){};

    virtual void init(const FlowStates<T, array_layout, host_mem_space>& fs,
                      FiniteVolume<T>& fv, const GridBlock<T>& grid,
                      const IdealGas<T>& gas_model,
                      const TransportProperties<T>& trans_prop) {
        (void)grid;
        (void)fs;
        (void)fv;
        (void)gas_model;
        (void)trans_prop;
    }

    virtual T access(const FlowStates<T, array_layout, host_mem_space>& fs,
                     FiniteVolume<T>& fv,
                     const GridBlock<T, host_exec_space, array_layout>& grid,
                     const IdealGas<T>& gas_model, const int i) = 0;
};

template <typename T>
class VectorAccessor {
public:
    virtual ~VectorAccessor(){};

    virtual void init(const FlowStates<T, array_layout, host_mem_space>& fs,
                      FiniteVolume<T>& fv, const GridBlock<T>& grid,
                      const IdealGas<T>& gas_model,
                      const TransportProperties<T>& trans_prop) {
        (void)grid;
        (void)fs;
        (void)fv;
        (void)gas_model;
        (void)trans_prop;
    }

    virtual Vector3<T> access(const FlowStates<T, array_layout, host_mem_space>& fs,
                              FiniteVolume<T>& fv,
                              const GridBlock<T, host_exec_space, array_layout>& grid,
                              const IdealGas<T>& gas_model, const int i) = 0;
};

template <typename T>
class PressureAccess : public ScalarAccessor<T> {
    T access(const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
             const GridBlock<T, host_exec_space, array_layout>& grid,
             const IdealGas<T>& gas_model, const int i);
};

template <typename T>
class TemperatureAccess : public ScalarAccessor<T> {
    T access(const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
             const GridBlock<T, host_exec_space, array_layout>& grid,
             const IdealGas<T>& gas_model, const int i);
};

template <typename T>
class DensityAccess : public ScalarAccessor<T> {
    T access(const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
             const GridBlock<T, host_exec_space, array_layout>& grid,
             const IdealGas<T>& gas_model, const int i);
};

template <typename T>
class InternalEnergyAccess : public ScalarAccessor<T> {
    T access(const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
             const GridBlock<T, host_exec_space, array_layout>& grid,
             const IdealGas<T>& gas_model, const int i);
};

template <typename T>
class SpeedOfSoundAccess : public ScalarAccessor<T> {
    T access(const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
             const GridBlock<T, host_exec_space, array_layout>& grid,
             const IdealGas<T>& gas_model, const int i);
};

template <typename T>
class MachNumberAccess : public ScalarAccessor<T> {
    T access(const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
             const GridBlock<T, host_exec_space, array_layout>& grid,
             const IdealGas<T>& gas_model, const int i);
};

template <typename T>
class VolumeAccess : public ScalarAccessor<T> {
    T access(const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
             const GridBlock<T, host_exec_space, array_layout>& grid,
             const IdealGas<T>& gas_model, const int i);
};

template <typename T>
class VelocityAccess : public VectorAccessor<T> {
    Vector3<T> access(const FlowStates<T, array_layout, host_mem_space>& fs,
                      FiniteVolume<T>& fv,
                      const GridBlock<T, host_exec_space, array_layout>& grid,
                      const IdealGas<T>& gas_model, const int i);
};

// viscous gradients
template <typename T>
class ViscousGradVxAccess : public VectorAccessor<T> {
public:
    void init(const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
              const GridBlock<T>& grid, const IdealGas<T>& gas_model,
              const TransportProperties<T>& trans_prop) override;

    Vector3<T> access(const FlowStates<T, array_layout, host_mem_space>& fs,
                      FiniteVolume<T>& fv,
                      const GridBlock<T, host_exec_space, array_layout>& grid,
                      const IdealGas<T>& gas_model, const int i) override;

private:
    Vector3s<T, array_layout, host_mem_space> grad_vx_;
};

template <typename T>
class ViscousGradVyAccess : public VectorAccessor<T> {
public:
    void init(const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
              const GridBlock<T>& grid, const IdealGas<T>& gas_model,
              const TransportProperties<T>& trans_prop) override;

    Vector3<T> access(const FlowStates<T, array_layout, host_mem_space>& fs,
                      FiniteVolume<T>& fv,
                      const GridBlock<T, host_exec_space, array_layout>& grid,
                      const IdealGas<T>& gas_model, const int i) override;

private:
    Vector3s<T, array_layout, host_mem_space> grad_vy_;
};

template <typename T>
class ViscousGradVzAccess : public VectorAccessor<T> {
public:
    void init(const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
              const GridBlock<T>& grid, const IdealGas<T>& gas_model,
              const TransportProperties<T>& trans_prop) override;

    Vector3<T> access(const FlowStates<T, array_layout, host_mem_space>& fs,
                      FiniteVolume<T>& fv,
                      const GridBlock<T, host_exec_space, array_layout>& grid,
                      const IdealGas<T>& gas_model, const int i) override;

private:
    Vector3s<T, array_layout, host_mem_space> grad_vz_;
};

// convective gradients
template <typename T>
class ConvectiveGradVxAccess : public VectorAccessor<T> {
public:
    void init(const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
              const GridBlock<T>& grid, const IdealGas<T>& gas_model,
              const TransportProperties<T>& trans_prop) override;

    Vector3<T> access(const FlowStates<T, array_layout, host_mem_space>& fs,
                      FiniteVolume<T>& fv,
                      const GridBlock<T, host_exec_space, array_layout>& grid,
                      const IdealGas<T>& gas_model, const int i) override;

private:
    Vector3s<T, array_layout, host_mem_space> grad_vx_;
};

template <typename T>
class ConvectiveGradVyAccess : public VectorAccessor<T> {
public:
    void init(const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
              const GridBlock<T>& grid, const IdealGas<T>& gas_model,
              const TransportProperties<T>& trans_prop) override;

    Vector3<T> access(const FlowStates<T, array_layout, host_mem_space>& fs,
                      FiniteVolume<T>& fv,
                      const GridBlock<T, host_exec_space, array_layout>& grid,
                      const IdealGas<T>& gas_model, const int i) override;

private:
    Vector3s<T, array_layout, host_mem_space> grad_vy_;
};

template <typename T>
class ConvectiveGradVzAccess : public VectorAccessor<T> {
public:
    void init(const FlowStates<T, array_layout, host_mem_space>& fs, FiniteVolume<T>& fv,
              const GridBlock<T>& grid, const IdealGas<T>& gas_model,
              const TransportProperties<T>& trans_prop) override;

    Vector3<T> access(const FlowStates<T, array_layout, host_mem_space>& fs,
                      FiniteVolume<T>& fv,
                      const GridBlock<T, host_exec_space, array_layout>& grid,
                      const IdealGas<T>& gas_model, const int i) override;

private:
    Vector3s<T, array_layout, host_mem_space> grad_vz_;
};

template <typename T>
class CellCentreAccess : public VectorAccessor<T> {
public:
    Vector3<T> access(const FlowStates<T, array_layout, host_mem_space>& fs,
                      FiniteVolume<T>& fv,
                      const GridBlock<T, host_exec_space, array_layout>& grid,
                      const IdealGas<T>& gas_model, const int i) override;
};

template <typename T>
std::map<std::string, std::shared_ptr<ScalarAccessor<T>>> get_scalar_accessors();

template <typename T>
std::map<std::string, std::shared_ptr<VectorAccessor<T>>> get_vector_accessors();

#endif
