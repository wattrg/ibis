#include <finite_volume/shock_fitting.h>

template <typename T>
ShockFitting<T>::ShockFitting(json config) {
    (void)config;
}

template <typename T>
void ShockFitting<T>::compute_vertex_velocities(const FlowStates<T>& fs,
                                                const GridBlock<T>& grid,
                                                Vector3s<T> vertex_vel) {
    (void)fs;
    (void)grid;
    (void)vertex_vel;
}
template class ShockFitting<Ibis::real>;
template class ShockFitting<Ibis::dual>;

template <typename T>
void ZeroVelocity<T>::apply(const FlowStates<T>& fs, const GridBlock<T>& grid,
                            Vector3s<T> vertex_vel,
                            const Field<size_t>& boundary_vertices) {
    (void)fs;
    (void)grid;
    Kokkos::parallel_for(
        "Shockfitting:zero_velocity", boundary_vertices.size(),
        KOKKOS_LAMBDA(const size_t i) {
            vertex_vel.x(i) = T(0.0);
            vertex_vel.y(i) = T(0.0);
            vertex_vel.z(i) = T(0.0);
        });
}
template class ZeroVelocity<Ibis::real>;
template class ZeroVelocity<Ibis::dual>;

template <typename T>
ConstrainDirection<T>::ConstrainDirection(json config) {
    Ibis::real x = config.at("x");
    Ibis::real y = config.at("y");
    Ibis::real z = config.at("z");
    direction_ = Vector3<T>(x, y, z);
}

template <typename T>
void ConstrainDirection<T>::apply(Vector3s<T> vertex_vel,
                                  Field<size_t>& boundary_vertices) {
    Vector3<T> dirn = direction_;
    Kokkos::parallel_for(
        "Shockfitting::ConstrainDirection", boundary_vertices.size(),
        KOKKOS_LAMBDA(const size_t i) {
            T vx = vertex_vel.x(i);
            T vy = vertex_vel.y(i);
            T vz = vertex_vel.z(i);
            T dot = dirn.x * vx + dirn.y * vy + dirn.z * vz;
            vertex_vel.x(i) = dirn.x * dot;
            vertex_vel.y(i) = dirn.y * dot;
            vertex_vel.z(i) = dirn.z * dot;
        });
}
template class ConstrainDirection<Ibis::real>;
template class ConstrainDirection<Ibis::dual>;

template <typename T>
ShockFittingInterpolationAction<T>::ShockFittingInterpolationAction(
    const GridBlock<T>& grid, std::vector<std::string> sample_markers,
    std::vector<std::string> interp_markers, Ibis::real power) {
    // get all the indices of the vertices to sample from
    std::vector<size_t> sample_points;
    for (std::string& marker_label : sample_markers) {
        Field<size_t> vertices = grid.marked_vertices(marker_label);
        for (size_t i = 0; i < vertices.size(); i++) {
            size_t vertex_i = vertices(i);
            if (std::find(sample_points.begin(), sample_points.end(), vertex_i) ==
                sample_points.end()) {
                // this point hasn't been seen before, so we'll add it
                sample_points.push_back(vertex_i);
            }
        }
    }
    sample_points_ = Field<size_t>("SFInterp::sample_points", sample_points);

    // get all the indices of the vertices to interpolate
    std::vector<size_t> interp_points;
    for (std::string& marker_label : interp_markers) {
        Field<size_t> vertices = grid.marked_vertices(marker_label);
        for (size_t i = 0; i < vertices.size(); i++) {
            size_t vertex_i = vertices(i);
            if (std::find(sample_points.begin(), sample_points.end(), vertex_i) !=
                sample_points.end()) {
                // this point already is a sample point
                continue;
            }
            if (std::find(interp_points.begin(), interp_points.end(), vertex_i) ==
                interp_points.end()) {
                // this point hasn't been seen before, so we'll add it
                interp_points.push_back(vertex_i);
            }
        }
    }
    interp_points_ = Field<size_t>("SFInterp::interp_points", interp_points);
    power_ = power;
}
