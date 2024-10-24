#ifndef SHOCK_FITTING_H
#define SHOCK_FITTING_H

#include <finite_volume/grid_motion_driver.h>

#include "Cuda/Kokkos_Cuda_Team.hpp"

template <typename T, class ExecSpace = Kokkos::DefaultExecutionSpace,
          class Layout = Kokkos::DefaultExecutionSpace::array_layout>
class ShockFitting : public GridMotionDriver<T, ExecSpace, Layout> {
public:
    using MemSpace = typename ExecSpace::memory_space;

public:
    ~ShockFitting() {}

    ShockFitting() {}

    void compute_vertex_velocities(const FlowStates<T, Layout, MemSpace>& fs,
                                   const GridBlock<T, ExecSpace, Layout>& grid,
                                   Vector3s<T, Layout, MemSpace> vertex_vel) {
        (void)fs;
        (void)grid;
        Kokkos::parallel_for(
            "shock_fitting", grid.num_vertices(), KOKKOS_LAMBDA(const size_t i) {
                vertex_vel.x(i) = T(1.0);
                vertex_vel.y(i) = T(0.0);
                vertex_vel.z(i) = T(0.0);
            });
    }
};

#endif
