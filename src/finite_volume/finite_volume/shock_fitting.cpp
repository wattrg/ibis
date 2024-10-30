#include <finite_volume/shock_fitting.h>
#include <spdlog/spdlog.h>

template <typename T>
ShockFitting<T>::ShockFitting(const GridBlock<T>& grid, json config) {
    std::vector<size_t> boundary_vertices;
    for (auto& [marker_label, config] : config.at("boundaries").items()) {
        // build the boundary condition
        bcs_.push_back(ShockFittingBC<T>(grid, marker_label, config));

        // keep track of the points on the boundaries, so that we can
        // interpolate the other points from them
        const Field<size_t> vertices = grid.marked_vertices(marker_label);
        for (size_t vertex_i = 0; vertex_i < vertices.size(); vertex_i++) {
            size_t vertex_id = vertices(vertex_i);
            if (std::find(boundary_vertices.begin(), boundary_vertices.end(),
                          vertex_id) == boundary_vertices.end()) {
                boundary_vertices.push_back(vertex_id);
            }
        }
    }

    // sort the vertices to interpolate (lets up use binary search later
    // for efficiency, and might help with coalescing memory access)
    std::sort(boundary_vertices.begin(), boundary_vertices.end());

    // get all the remaining points which must be interpolated
    std::vector<size_t> internal_vertices;
    for (size_t vertex_id = 0; vertex_id < grid.num_vertices(); vertex_id++) {
        if (!std::binary_search(boundary_vertices.begin(), boundary_vertices.end(),
                                vertex_id)) {
            // this vertex is not on a boundary
            internal_vertices.push_back(vertex_id);
        }
    }

    Field<size_t> sample_points{"ShockFit::sample_points", boundary_vertices};
    Field<size_t> interp_points{"ShockFit::interp_points", internal_vertices};
    Ibis::real power = config.at("interp_power");
    interp_ = ShockFittingInterpolationAction<T>(sample_points, interp_points, power);
}

template <typename T>
void ShockFitting<T>::compute_vertex_velocities(const FlowStates<T>& fs,
                                                const GridBlock<T>& grid,
                                                Vector3s<T> vertex_vel) {
    // Step 1: Compute velocities of vertices which have direct equations
    for (auto& bc : bcs_) {
        bc.apply_direct_actions(fs, grid, vertex_vel);
    }

    // Step 2: Interpolate velocities of vertices on boundaries
    for (auto& bc : bcs_) {
        bc.apply_interp_actions(grid, vertex_vel);
    }

    // Step 3: Constrain certain vertices to move in given direction
    for (auto& bc : bcs_) {
        bc.apply_constraints(grid, vertex_vel);
    }

    // Step 4: Interpolate the remaining internal velocities
    interp_.apply(grid, vertex_vel);
}
template class ShockFitting<Ibis::real>;
template class ShockFitting<Ibis::dual>;

template <typename T>
ShockFittingBC<T>::ShockFittingBC(const GridBlock<T>& grid, std::string marker,
                                  json config) {
    json direct_configs = config.at("direct");
    for (json direct_config : direct_configs) {
        direct_actions_.push_back(
            {marker, make_direct_velocity_action<T>(direct_config)});
    }

    json interp_configs = config.at("interp");
    for (json interp_config : interp_configs) {
        interp_actions_.push_back(
            ShockFittingInterpolationAction<T>(grid, marker, interp_config));
    }

    json constraint_configs = config.at("constraint");
    for (json constraint_config : constraint_configs) {
        constraints_.push_back({marker, ConstrainDirection<T>(constraint_config)});
    }
}

template <typename T>
void ShockFittingBC<T>::apply_direct_actions(const FlowStates<T>& fs,
                                             const GridBlock<T>& grid,
                                             Vector3s<T> vertex_vel) {
    for (auto& [marker, action] : direct_actions_) {
        const Field<size_t>& vertices = grid.marked_vertices(marker);
        action->apply(fs, grid, vertex_vel, vertices);
    }
}

template <typename T>
void ShockFittingBC<T>::apply_interp_actions(const GridBlock<T>& grid,
                                             Vector3s<T> vertex_vel) {
    for (auto& action : interp_actions_) {
        action.apply(grid, vertex_vel);
    }
}

template <typename T>
void ShockFittingBC<T>::apply_constraints(const GridBlock<T>& grid,
                                          Vector3s<T> vertex_vel) {
    for (auto& [marker, action] : constraints_) {
        const Field<size_t>& vertices = grid.marked_vertices(marker);
        action.apply(vertex_vel, vertices);
    }
}
template class ShockFittingBC<Ibis::real>;
template class ShockFittingBC<Ibis::dual>;

template <typename T>
FixedVelocity<T>::FixedVelocity(json config) {
    json velocity = config.at("velocity");
    T x = T(velocity.at("x"));
    T y = T(velocity.at("y"));
    T z = T(velocity.at("z"));
    vel_ = Vector3<T>(x, y, z);
}

template <typename T>
void FixedVelocity<T>::apply(const FlowStates<T>& fs, const GridBlock<T>& grid,
                             Vector3s<T> vertex_vel,
                             const Field<size_t>& boundary_vertices) {
    (void)fs;
    (void)grid;
    Vector3<T> vel = vel_;
    Kokkos::parallel_for(
        "Shockfitting:zero_velocity", boundary_vertices.size(),
        KOKKOS_LAMBDA(const size_t i) {
            vertex_vel.x(i) = T(vel.x);
            vertex_vel.y(i) = T(vel.y);
            vertex_vel.z(i) = T(vel.z);
        });
}
template class FixedVelocity<Ibis::real>;
template class FixedVelocity<Ibis::dual>;

template <typename T>
WaveSpeed<T>::WaveSpeed(json config) {
    scale_ = config.at("scale");
}

template <typename T>
void WaveSpeed<T>::apply(const FlowStates<T>& fs, const GridBlock<T>& grid,
                         Vector3s<T> vertex_vel, const Field<size_t>& boundary_vertices) {
    (void)fs;
    (void)grid;
    (void)vertex_vel;
    (void)boundary_vertices;
}
template class WaveSpeed<Ibis::real>;
template class WaveSpeed<Ibis::dual>;

template <typename T>
std::shared_ptr<ShockFittingDirectVelocityAction<T>> make_direct_velocity_action(
    json config) {
    std::string type = config.at("type");
    if (type == "shock_fit") {
        return std::shared_ptr<ShockFittingDirectVelocityAction<T>>(
            new WaveSpeed<T>(config));
    } else if (type == "fixed_velocity") {
        return std::shared_ptr<ShockFittingDirectVelocityAction<T>>(
            new FixedVelocity<T>(config));
    } else {
        spdlog::error("Unkown grid motion direct velocity action {}", type);
        throw new std::runtime_error("Unkown grid motion direction velocity action");
    }
}
template std::shared_ptr<ShockFittingDirectVelocityAction<Ibis::real>>
    make_direct_velocity_action(json);
template std::shared_ptr<ShockFittingDirectVelocityAction<Ibis::dual>>
    make_direct_velocity_action(json);

template <typename T>
ConstrainDirection<T>::ConstrainDirection(json config) {
    Ibis::real x = config.at("x");
    Ibis::real y = config.at("y");
    Ibis::real z = config.at("z");
    direction_ = Vector3<T>(x, y, z);
}

template <typename T>
void ConstrainDirection<T>::apply(Vector3s<T> vertex_vel,
                                  const Field<size_t>& boundary_vertices) {
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
    std::string interp_marker, Ibis::real power) {
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
    Field<size_t> vertices = grid.marked_vertices(interp_marker);
    for (size_t i = 0; i < vertices.size(); i++) {
        size_t vertex_i = vertices(i);
        if (std::find(sample_points.begin(), sample_points.end(), vertex_i) !=
            sample_points.end()) {
            // this point is already is a sample point
            continue;
        }
        interp_points.push_back(vertex_i);
    }
    interp_points_ = Field<size_t>("SFInterp::interp_points", interp_points);
    power_ = power;
}

template <typename T>
ShockFittingInterpolationAction<T>::ShockFittingInterpolationAction(
    const GridBlock<T>& grid, std::string interp_marker, json config)
    : ShockFittingInterpolationAction(grid, config.at("sample_points"), interp_marker,
                                      config.at("power")) {}

template <typename T>
void ShockFittingInterpolationAction<T>::apply(const GridBlock<T>& grid,
                                               Vector3s<T> vertex_vel) {
    // Inverse distance weighting as per
    // https://en.wikipedia.org/wiki/Inverse_distance_weighting
    Field<size_t> sample_points = sample_points_;
    size_t num_sample_points = sample_points.size();
    Field<size_t> interp_points = interp_points_;
    Vertices<T> vertices = grid.vertices();
    T power = T(power_);
    Kokkos::parallel_for(
        "SFInterp::interp", interp_points.size(), KOKKOS_LAMBDA(const size_t interp_i) {
            size_t interp_id = interp_points(interp_i);
            Vector3<T> num;
            T den = T(0.0);
            Vector3<T> interp_pos = vertices.position(interp_id);
            for (size_t sample_i = 0; sample_i < num_sample_points; sample_i++) {
                size_t sample_id = sample_points(sample_i);
                Vector3<T> sample_pos = vertices.position(sample_id);
                T dx = Ibis::abs(sample_pos.x - interp_pos.x);
                T dy = Ibis::abs(sample_pos.y - interp_pos.y);
                T dz = Ibis::abs(sample_pos.z - interp_pos.z);
                T dis = Ibis::sqrt(dx * dx + dy * dy + dz * dz);
                T w = 1.0 / Ibis::pow(dis, power);
                num.x += w * vertex_vel.x(sample_id);
                num.y += w * vertex_vel.y(sample_id);
                num.z += w * vertex_vel.z(sample_id);
                den += w;
            }
            vertex_vel.x(interp_id) = num.x / den;
            vertex_vel.y(interp_id) = num.y / den;
            vertex_vel.z(interp_id) = num.z / den;
        });
}

template class ShockFittingInterpolationAction<Ibis::real>;
template class ShockFittingInterpolationAction<Ibis::dual>;
