#ifndef RIGID_BODY_TRANSLATION_H
#define RIGID_BODY_TRANSLATION_H

#include <finite_volume/grid_motion_driver.h>

template <typename T, class MemModel>
class RigidBodyTranslation : public GridMotionDriver<T, MemModel> {
public:
    ~RigidBodyTranslation() {}

    RigidBodyTranslation() {}

    RigidBodyTranslation(json config);

    void compute_vertex_velocities(const FlowStates<T>& fs,
                                   const GridBlock<MemModel, T>& grid,
                                   Vector3s<T> vertex_vel);

private:
    Vector3<T> vel_;
};

#endif
