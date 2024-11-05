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
        auto vertices = grid.marked_vertices(marker_label).host_mirror();
        vertices.deep_copy(grid.marked_vertices(marker_label));
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
            {marker, make_direct_velocity_action<T>(grid, marker, direct_config)});
    }

    json interp_configs = config.at("interp");
    for (json interp_config : interp_configs) {
        interp_actions_.push_back(
            ShockFittingInterpolationAction<T>(grid, marker, interp_config));
    }

    json constraint_configs = config.at("constraint");
    for (json constraint_config : constraint_configs) {
        constraints_.push_back({marker, make_constraint<T>(constraint_config)});
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
        action->apply(grid, vertex_vel, vertices);
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
            size_t vertex_i = boundary_vertices(i);
            vertex_vel.x(vertex_i) = T(vel.x);
            vertex_vel.y(vertex_i) = T(vel.y);
            vertex_vel.z(vertex_i) = T(vel.z);
        });
}
template class FixedVelocity<Ibis::real>;
template class FixedVelocity<Ibis::dual>;

template <typename T>
WaveSpeed<T>::WaveSpeed(const GridBlock<T>& grid, std::string marker, json config) {
    scale_ = config.at("scale");
    shock_detection_threshold_ = config.at("shock_detection_threshold");
    constraint_ = make_constraint<T>(config.at("constraint"));

    // the position of each marked vertex in the array of marked vertices
    auto marked_vertices = grid.marked_vertices(marker).host_mirror();
    marked_vertices.deep_copy(grid.marked_vertices(marker));
    std::map<size_t, size_t> marked_vertex_index;
    for (size_t vertex_index = 0; vertex_index < marked_vertices.size(); vertex_index++) {
        size_t vertex_id = marked_vertices(vertex_index);
        marked_vertex_index.insert({vertex_id, vertex_index});
    }

    // which faces attached to the vertices should be shock fit
    std::vector<std::vector<size_t>> shock_fit_faces_connectivity(marked_vertices.size());
    // shock_fit_faces_connectivity.reserve(marked_vertices.size());

    // the faces to shockfit
    auto marked_faces = grid.marked_faces(marker).host_mirror();
    marked_faces.deep_copy(grid.marked_faces(marker));

    // vertices of each face
    auto face_vertices = grid.interfaces().vertex_ids().host_mirror();
    face_vertices.deep_copy(grid.interfaces().vertex_ids());

    for (size_t face_i = 0; face_i < marked_faces.size(); face_i++) {
        size_t face_id = marked_faces(face_i);
        auto face_id_vertices = face_vertices(face_id);
        for (size_t vertex_i = 0; vertex_i < face_id_vertices.size(); vertex_i++) {
            size_t vertex_id = face_id_vertices(vertex_i);
            size_t vertex_marker_index = marked_vertex_index[vertex_id];
            shock_fit_faces_connectivity[vertex_marker_index].push_back(face_id);
        }
    }

    faces_ = Ibis::RaggedArray<size_t>{shock_fit_faces_connectivity};
}

template <typename T>
KOKKOS_INLINE_FUNCTION T wave_speed(const FlowState<T>& left, const FlowState<T>& right,
                                    const Vector3<T>& norm,
                                    Ibis::real shock_detection_threshold) {
    // the face normal
    T nx = norm.x;
    T ny = norm.y;
    T nz = norm.z;

    // left gas state
    T pL = left.gas_state.pressure;
    T rL = left.gas_state.rho;
    T uL = left.velocity.x * nx + left.velocity.y * ny + left.velocity.z * nz;

    // right gas state
    T pR = right.gas_state.pressure;
    T rR = right.gas_state.rho;
    T uR = right.velocity.x * nx + right.velocity.y * ny + right.velocity.z * nz;

    // shock detector
    T delta_rho = Ibis::abs(rL - rR);
    T max_rho = Ibis::max(rL, rR);
    bool shock_detected = (delta_rho / max_rho) > shock_detection_threshold;

    if (shock_detected) {
        // wave speed from the mass conservation equation
        T ws1 = (rR * uR - rL * uL) / (rL - rR);

        // wave speeds from normal momentum conservation equation
        T a = pL - pR + rL * uL * uL - rR * uR * uR;
        T b = T(2.0) * (rR * uR - rL * uL);
        T c = rL - rR;
        int sign = (pR - pL > T(0.0)) ? -1 : 1;
        T ws2 = (-b + sign * Ibis::sqrt(b * b - 4 * a * c)) / (T(2.0) * a);
        return 0.5 * ws1 + 0.5 * ws2;
    } else {
        // assume ideal gas for the moment. Will have to pass a gas model in
        // at some point
        T aL = Ibis::sqrt(1.4 * 287.0 * left.gas_state.temp);
        T aR = Ibis::sqrt(1.4 * 287.0 * right.gas_state.temp);

        T ML = uL / aL;
        T MR = uR / aR;
        T wL = (ML + Ibis::abs(ML)) / 2;
        T wR = (MR - Ibis::abs(MR)) / 2;
        return wL / (wL + wR) * (uL - aL) + wR / (wL + wR) * (uR + aR);
    }
}

template <typename T>
KOKKOS_FUNCTION T mach_weighting(const FlowState<T>& left, const FlowState<T>& right,
                                 const Vector3<T>& vertex_pos, const Vector3<T>& face_pos,
                                 const Vector3<T>& face_norm) {
    // determine velocity at the interface
    T uL = left.velocity.x * face_norm.x + left.velocity.y * face_norm.y +
           left.velocity.z * face_norm.z;
    T uR = right.velocity.x * face_norm.x + right.velocity.y * face_norm.y +
           right.velocity.z * face_norm.z;
    T aL = Ibis::sqrt(1.4 * 287.0 * left.gas_state.temp);
    T aR = Ibis::sqrt(1.4 * 287.0 * right.gas_state.temp);
    T ML = uL / aL;
    T MR = uR / aR;
    T wL = (ML + Ibis::abs(ML)) / 2;
    T wR = (MR - Ibis::abs(MR)) / 2;
    FlowState<T> face_fs;
    face_fs.set_weighted_average(left, wL / (wL + wR), right, wR / (wL + wR));

    T tan_x = vertex_pos.x - face_pos.x;
    T tan_y = vertex_pos.y - face_pos.y;
    T tan_z = vertex_pos.z - face_pos.z;
    T len_tan = Ibis::sqrt(tan_x * tan_x + tan_y * tan_y + tan_z * tan_z);
    T face_a = Ibis::sqrt(1.4 * 287 * face_fs.gas_state.temp);
    T face_mach = (face_fs.velocity.x * tan_x + face_fs.velocity.y * tan_y +
                   face_fs.velocity.z * tan_z) /
                  (face_a * len_tan);
    return (face_mach + Ibis::abs(face_mach)) / 2;
}

template <typename T>
void WaveSpeed<T>::apply(const FlowStates<T>& fs, const GridBlock<T>& grid,
                         Vector3s<T> vertex_vel, const Field<size_t>& boundary_vertices) {
    (void)fs;
    (void)grid;
    (void)vertex_vel;
    (void)boundary_vertices;
    Ibis::real scale = scale_;
    Ibis::real shock_detection_threshold = shock_detection_threshold_;
    auto interface_ids = grid.vertices().interface_ids();
    auto normals = grid.interfaces().norm();
    auto vertices = grid.vertices();
    auto grid_interfaces = grid.interfaces();
    auto face_connectivity = faces_;
    Kokkos::parallel_for(
        "ShockFitting::wave_speed", boundary_vertices.size(),
        KOKKOS_LAMBDA(const size_t i) {
            size_t vertex_i = boundary_vertices(i);
            auto interfaces = face_connectivity(i);
            // we repeat the calculation of the wave speed for each face for
            // each vertex attached to the face. This is a bit wasteful of compute,
            // but requires less memory. Some profiling would be good to see which
            // is faster overall
            T num_x = T(0.0);
            T num_y = T(0.0);
            T num_z = T(0.0);
            T num_unweighted_x = T(0.0);
            T num_unweighted_y = T(0.0);
            T num_unweighted_z = T(0.0);
            T den = T(0.0);
            Vector3<T> vertex_pos = vertices.position(vertex_i);
            for (size_t face_i = 0; face_i < interfaces.size(); face_i++) {
                size_t face_id = interfaces(face_i);
                size_t left_id = grid_interfaces.left_cell(face_id);
                size_t right_id = grid_interfaces.right_cell(face_id);
                FlowState<T> left = fs.flow_state(left_id);
                FlowState<T> right = fs.flow_state(right_id);
                Vector3<T> norm = normals.vector(face_id);
                T ws = wave_speed(left, right, norm, shock_detection_threshold);

                // T weight = T(1.0);
                Vector3<T> face_pos = grid_interfaces.centre().vector(face_id);
                T weight = mach_weighting(left, right, vertex_pos, face_pos, norm);
                // printf("ws = %.16f, weight = %.16f\n", ws, weight);
                num_x += weight * ws * norm.x;
                num_y += weight * ws * norm.y;
                num_z += weight * ws * norm.z;
                num_unweighted_x += ws * norm.x;
                num_unweighted_y += ws * norm.y;
                num_unweighted_z += ws * norm.z;
                den += weight;
                // printf("face %ld, ws = %.16f, norm = [%.16f, %.16f, %.16f]\n", face_id,
                // ws, norm.x, norm.y, norm.z);
            }
            // printf("\n");
            if (den < 1e-14) {
                vertex_vel.x(vertex_i) = num_unweighted_x / interfaces.size() * scale;
                vertex_vel.y(vertex_i) = num_unweighted_y / interfaces.size() * scale;
                vertex_vel.z(vertex_i) = num_unweighted_z / interfaces.size() * scale;
            } else {
                vertex_vel.x(vertex_i) = num_x / den * scale;
                vertex_vel.y(vertex_i) = num_y / den * scale;
                vertex_vel.z(vertex_i) = num_z / den * scale;
            }
        });

    if (constraint_) {
        constraint_->apply(grid, vertex_vel, boundary_vertices);
    }
}
template class WaveSpeed<Ibis::real>;
template class WaveSpeed<Ibis::dual>;

template <typename T>
ConstrainDirection<T>::ConstrainDirection(json config) {
    json direction = config.at("direction");
    Ibis::real x = direction.at("x");
    Ibis::real y = direction.at("y");
    Ibis::real z = direction.at("z");
    direction_ = Vector3<T>(x, y, z);
}

template <typename T>
void ConstrainDirection<T>::apply(const GridBlock<T>& grid, Vector3s<T> vertex_vel,
                                  const Field<size_t>& boundary_vertices) {
    (void)grid;
    Vector3<T> dirn = direction_;
    Kokkos::parallel_for(
        "Shockfitting::ConstrainDirection", boundary_vertices.size(),
        KOKKOS_LAMBDA(const size_t i) {
            size_t vertex_i = boundary_vertices(i);
            T vx = vertex_vel.x(vertex_i);
            T vy = vertex_vel.y(vertex_i);
            T vz = vertex_vel.z(vertex_i);
            T dot = dirn.x * vx + dirn.y * vy + dirn.z * vz;
            vertex_vel.x(vertex_i) = dirn.x * dot;
            vertex_vel.y(vertex_i) = dirn.y * dot;
            vertex_vel.z(vertex_i) = dirn.z * dot;
        });
}
template class ConstrainDirection<Ibis::real>;
template class ConstrainDirection<Ibis::dual>;

template <typename T>
RadialConstraint<T>::RadialConstraint(json config) {
    json centre = config.at("centre");
    Ibis::real x = centre.at("x");
    Ibis::real y = centre.at("y");
    Ibis::real z = centre.at("z");
    centre_ = Vector3<T>{x, y, z};
}

template <typename T>
void RadialConstraint<T>::apply(const GridBlock<T>& grid, Vector3s<T> vertex_vel,
                                const Field<size_t>& boundary_vertices) {
    Vector3<T> centre = centre_;
    Vector3s<T> vertex_pos = grid.vertices().positions();
    Kokkos::parallel_for(
        "Shockfitting::RadialConstraint", boundary_vertices.size(),
        KOKKOS_LAMBDA(const size_t i) {
            size_t vertex_i = boundary_vertices(i);
            Vector3<T> pos = vertex_pos.vector(vertex_i);
            Vector3<T> dirn =
                Vector3<T>{centre.x - pos.x, centre.y - pos.y, centre.z - pos.z};
            T len_dir = Ibis::sqrt(dirn.x * dirn.x + dirn.y * dirn.y + dirn.z * dirn.z);
            dirn.x /= len_dir;
            dirn.y /= len_dir;
            dirn.z /= len_dir;
            Vector3<T> vel = vertex_vel.vector(vertex_i);
            T dot = dirn.x * vel.x + dirn.y * vel.y + dirn.z * vel.z;
            vertex_vel.x(vertex_i) = dirn.x * dot;
            vertex_vel.y(vertex_i) = dirn.y * dot;
            vertex_vel.z(vertex_i) = dirn.z * dot;
        });
}
template class RadialConstraint<Ibis::real>;
template class RadialConstraint<Ibis::dual>;

template <typename T>
ShockFittingInterpolationAction<T>::ShockFittingInterpolationAction(
    const GridBlock<T>& grid, std::vector<std::string> sample_markers,
    std::string interp_marker, Ibis::real power) {
    // get all the indices of the vertices to sample from
    std::vector<size_t> sample_points;
    for (std::string& marker_label : sample_markers) {
        auto vertices = grid.marked_vertices(marker_label).host_mirror();
        vertices.deep_copy(grid.marked_vertices(marker_label));
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
    auto vertices = grid.marked_vertices(interp_marker).host_mirror();
    vertices.deep_copy(grid.marked_vertices(interp_marker));
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

template <typename T>
std::shared_ptr<ShockFittingDirectVelocityAction<T>> make_direct_velocity_action(
    const GridBlock<T>& grid, std::string marker, json config) {
    std::string type = config.at("type");
    if (type == "wave_speed") {
        return std::shared_ptr<ShockFittingDirectVelocityAction<T>>(
            new WaveSpeed<T>(grid, marker, config));
    } else if (type == "fixed_velocity") {
        return std::shared_ptr<ShockFittingDirectVelocityAction<T>>(
            new FixedVelocity<T>(config));
    } else {
        spdlog::error("Unkown grid motion direct velocity action {}", type);
        throw new std::runtime_error("Unkown grid motion direction velocity action");
    }
}
template std::shared_ptr<ShockFittingDirectVelocityAction<Ibis::real>>
make_direct_velocity_action(const GridBlock<Ibis::real>&, std::string, json);
template std::shared_ptr<ShockFittingDirectVelocityAction<Ibis::dual>>
make_direct_velocity_action(const GridBlock<Ibis::dual>&, std::string, json);

template <typename T>
std::shared_ptr<Constraint<T>> make_constraint(json config) {
    std::string type = config.at("type");
    if (type == "direction") {
        return std::shared_ptr<Constraint<T>>(new ConstrainDirection<T>(config));
    } else if (type == "radial") {
        return std::shared_ptr<Constraint<T>>(new RadialConstraint<T>(config));
    } else if (type == "none") {
        return std::shared_ptr<Constraint<T>>();
    } else {
        spdlog::error("Unknown grid motion constraint {}", type);
        throw new std::runtime_error("Unkown grid motion constraint");
    }
}
template std::shared_ptr<Constraint<Ibis::real>> make_constraint(json);
template std::shared_ptr<Constraint<Ibis::dual>> make_constraint(json);
