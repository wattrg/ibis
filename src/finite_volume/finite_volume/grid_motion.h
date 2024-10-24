#ifndef GRID_MOTION_H
#define GRID_MOTION_H

#include <finite_volume/grid_motion_driver.h>
#include <gas/flow_state.h>
#include <grid/grid.h>

template <typename T, class ExecSpace = Kokkos::DefaultExecutionSpace,
          class Layout = Kokkos::DefaultExecutionSpace::array_layout>
class GridMotion {
public:
    using MemSpace = typename ExecSpace::memory_space;

public:
    GridMotion() {}

    void compute_grid_motion(const FlowStates<T, Layout, MemSpace>& fs,
                             const GridBlock<T, ExecSpace, Layout>& grid) {
        driver_.compute_vertex_velocities(fs, grid, vertex_vel_);
        compute_face_vel(grid);
    }

    void compute_face_vel(const GridBlock<T, ExecSpace, Layout>& grid) {
        auto face_vertices = grid.interfaces().vertex_ids();
        Kokkos::parallel_for("compute_face_velocity", grid.num_interfaces(),
            KOKKOS_LAMBDA(const size_t face_i) {
            auto vertices = face_vertices(face_i);
            T vx = T(0.0);
            T vy = T(0.0);
            T vz = T(0.0);
            size_t num_vertices = vertices.size();
            for (size_t vertex_i = 0; vertex_i < num_vertices; vertex_i++) {
                size_t vertex_id = vertices(vertex_i);
                vx += vertex_vel_.x(vertex_id);
                vy += vertex_vel_.y(vertex_id);
                vz += vertex_vel_.z(vertex_id);
            }
            face_vel_.x(face_i) = vx / num_vertices;
            face_vel_.y(face_i) = vy / num_vertices;
            face_vel_.z(face_i) = vz / num_vertices;
        });
    }

private:
    Vector3s<T, Layout, MemSpace> vertex_vel_;
    Vector3s<T, Layout, MemSpace> face_vel_;
    GridMotionDriver<T, ExecSpace, Layout> driver_;
};

#endif
