#ifndef SHOCK_FITTING_H
#define SHOCK_FITTING_H

#include <finite_volume/grid_motion_driver.h>

template <typename T>
class ShockFitting : public GridMotionDriver<T> {
public:
    ~ShockFitting() {}

    ShockFitting() {}

    ShockFitting(json config);

    void compute_vertex_velocities(const FlowStates<T>& fs, const GridBlock<T>& grid,
                                   Vector3s<T> vertex_vel);
};

#endif
