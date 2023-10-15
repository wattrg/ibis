#ifndef ACCESSOR_H
#define ACCESSOR_H

#include <map>
#include <Kokkos_Core.hpp>
#include "../../src/gas/src/flow_state.h"

template <typename T>
class ScalarAccessor{
public:
    virtual ~ScalarAccessor(){};
    virtual T access(const FlowStates<T>& fs, const int i) = 0;
};


template <typename T>
class PressureAccess : public ScalarAccessor<T>{
    T access (const FlowStates<T>& fs, const int i);
};

template <typename T>
class TemperatureAccess : public ScalarAccessor<T>{
    T access (const FlowStates<T>& fs, const int i);
};

template <typename T>
class DensityAccess : public ScalarAccessor<T>{
    T access (const FlowStates<T>& fs, const int i);
};

template <typename T>
class InternalEnergyAccess : public ScalarAccessor<T> {
    T access (const FlowStates<T>& fs, const int i);
};

template <typename T>
class SpeedOfSoundAccess : public ScalarAccessor<T> {
    T access (const FlowStates<T>& fs, const int i);
};

template <typename T>
class MachNumberAccess : public ScalarAccessor<T> {
    T access (const FlowStates<T>& fs, const int i);
};


template <typename T>
std::map<std::string, std::shared_ptr<ScalarAccessor<T>>> get_accessors();

#endif
