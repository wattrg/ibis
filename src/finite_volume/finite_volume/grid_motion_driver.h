#ifndef GRID_MOTION_DRIVER_H
#define GRID_MOTION_DRIVER_H

#include <gas/flow_state.h>
#include <grid/grid.h>

template <typename T>
class GridMotionDriver {
public:
    virtual ~GridMotionDriver() {}

    GridMotionDriver() {}

    virtual void compute_vertex_velocities(const FlowStates<T>& fs,
                                           const GridBlock<T>& grid,
                                           Vector3s<T> vertex_vel) = 0;
};

template <typename T>
std::shared_ptr<GridMotionDriver<T>> build_grid_motion_driver(json config);

#endif
