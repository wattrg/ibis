#ifndef GRID_MOTION_H
#define GRID_MOTION_H

// #include <finite_volume/grid_motion_driver.h>
#include <gas/flow_state.h>
#include <grid/grid.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

template <typename T, class ExecSpace, class Layout>
class GridMotionDriver;

template <typename T, class ExecSpace = Kokkos::DefaultExecutionSpace,
          class Layout = Kokkos::DefaultExecutionSpace::array_layout>
class GridMotion {
public:
    using MemSpace = typename ExecSpace::memory_space;

public:
    GridMotion() {}

    GridMotion(const GridBlock<T, ExecSpace, Layout>& grid, json config) {
        enabled_ = config.at("enabled");
        if (enabled_) {
            face_vel_ = Vector3s<T, Layout, MemSpace>(grid.num_interfaces());
        }
    }

    void compute_grid_motion(const FlowStates<T, Layout, MemSpace>& fs,
                             const GridBlock<T, ExecSpace, Layout>& grid,
                             std::shared_ptr<GridMotionDriver<T, ExecSpace, Layout>>& driver,
                             Vector3s<T, Layout, MemSpace> vertex_vel) {
        driver->compute_vertex_velocities(fs, grid, vertex_vel);
        compute_face_vel(grid, vertex_vel);
    }

    void compute_face_vel(const GridBlock<T, ExecSpace, Layout>& grid,
                          const Vector3s<T, Layout, MemSpace>& vertex_vel) {
        auto face_vertices = grid.interfaces().vertex_ids();
        Kokkos::parallel_for(
            "compute_face_velocity", grid.num_interfaces(),
            KOKKOS_LAMBDA(const size_t face_i) {
                auto vertices = face_vertices(face_i);
                T vx = T(0.0);
                T vy = T(0.0);
                T vz = T(0.0);
                size_t num_vertices = vertices.size();
                for (size_t vertex_i = 0; vertex_i < num_vertices; vertex_i++) {
                    size_t vertex_id = vertices(vertex_i);
                    vx += vertex_vel.x(vertex_id);
                    vy += vertex_vel.y(vertex_id);
                    vz += vertex_vel.z(vertex_id);
                }
                face_vel_.x(face_i) = vx / num_vertices;
                face_vel_.y(face_i) = vy / num_vertices;
                face_vel_.z(face_i) = vz / num_vertices;
            });
    }

    const Vector3s<T, Layout, MemSpace> face_vel() { return face_vel_; }

    bool enabled() { return enabled_; }

private:
    bool enabled_;
    Vector3s<T, Layout, MemSpace> face_vel_;
};

#endif
