#ifndef GRID_MOTION_DRIVER_H
#define GRID_MOTION_DRIVER_H

#include <gas/flow_state.h>
#include <grid/grid.h>

template <typename T, class ExecSpace = Kokkos::DefaultExecutionSpace,
          class Layout = Kokkos::DefaultExecutionSpace::array_layout>
class GridMotionDriver {
public:
    using MemSpace = typename ExecSpace::memory_space;

public:
    virtual ~GridMotionDriver() {}

    GridMotionDriver() {}

    virtual void compute_vertex_velocities(const FlowStates<T, Layout, MemSpace>& fs,
                                           const GridBlock<T, ExecSpace, Layout>& grid,
                                           Vector3s<T, Layout, MemSpace> vertex_vel) = 0;
};

#endif
