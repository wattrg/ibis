#ifndef SHOCK_FITTING_H
#define SHOCK_FITTING_H

#include <finite_volume/grid_motion_driver.h>

#include "util/ragged_array.h"

template <typename T, class MemModel>
class Constraint {
public:
    virtual ~Constraint() {}

    virtual void apply(const GridBlock<MemModel, T>& grid, Vector3s<T> vertex_vel,
                       const Field<size_t>& vertices) = 0;
};

template <typename T, class MemModel>
class ConstrainDirection : public Constraint<T, MemModel> {
public:
    ~ConstrainDirection() {}

    ConstrainDirection(Ibis::real x, Ibis::real y, Ibis::real z)
        : direction_(Vector3<T>(x, y, z)) {}

    ConstrainDirection(json config);

    void apply(const GridBlock<MemModel, T>& grid, Vector3s<T> vertex_vel,
               const Field<size_t>& boundary_vertices);

private:
    Vector3<T> direction_;
};

template <typename T, class MemModel>
class RadialConstraint : public Constraint<T, MemModel> {
public:
    ~RadialConstraint() {}

    RadialConstraint(Vector3<T> centre) : centre_(centre) {}

    RadialConstraint(json config);

    void apply(const GridBlock<MemModel, T>& grid, Vector3s<T> vertex_vel,
               const Field<size_t>& boundary_vertices);

private:
    Vector3<T> centre_;
};

template <typename T, class MemModel>
class ShockFittingDirectVelocityAction {
public:
    virtual ~ShockFittingDirectVelocityAction() {}

    virtual void apply(const FlowStates<T>& fs, const GridBlock<MemModel, T>& grid,
                       Vector3s<T> vertex_vel,
                       const Field<size_t>& boundary_vertices) = 0;
};

template <typename T, class MemModel>
class WaveSpeed : public ShockFittingDirectVelocityAction<T, MemModel> {
public:
    ~WaveSpeed() {}

    WaveSpeed() {}

    WaveSpeed(const GridBlock<MemModel, T>& grid, std::string marker, json config);

    void apply(const FlowStates<T>& fs, const GridBlock<MemModel, T>& grid, Vector3s<T> vertex_vel,
               const Field<size_t>& boundary_vertices);

private:
    Ibis::real scale_;
    Ibis::real shock_detection_threshold_;
    Ibis::real shock_detection_width_;
    std::shared_ptr<Constraint<T, MemModel>> constraint_;
    Ibis::RaggedArray<size_t> faces_;
    // FlowState<T> flow_state_;
};

template <typename T, class MemModel>
class FixedVelocity : public ShockFittingDirectVelocityAction<T, MemModel> {
public:
    ~FixedVelocity() {}

    FixedVelocity() {}

    FixedVelocity(Vector3<T> vel) : vel_(vel) {}

    FixedVelocity(json config);

    void apply(const FlowStates<T>& fs, const GridBlock<MemModel, T>& grid, Vector3s<T> vertex_vel,
               const Field<size_t>& boundary_vertices);

private:
    Vector3<T> vel_;
};

template <typename T, class MemModel>
class ShockFittingInterpolationAction {
public:
    ~ShockFittingInterpolationAction() {}

    ShockFittingInterpolationAction() {}

    ShockFittingInterpolationAction(Field<size_t> sample_points,
                                    Field<size_t> interp_points, Ibis::real power)
        : sample_points_(sample_points), interp_points_(interp_points), power_(power) {}

    ShockFittingInterpolationAction(const GridBlock<MemModel, T>& grid,
                                    std::vector<std::string> sample_markers,
                                    std::string interp_marker, Ibis::real power);

    ShockFittingInterpolationAction(const GridBlock<MemModel, T>& grid, std::string interp_marker,
                                    json config);

    void apply(const GridBlock<MemModel, T>& grid, Vector3s<T> vertex_vel);

private:
    Field<size_t> sample_points_;
    Field<size_t> interp_points_;
    Ibis::real power_;
};

template <typename T, class MemModel>
class ShockFittingBC {
public:
    ShockFittingBC() {}

    ShockFittingBC(const GridBlock<MemModel, T>& grid, std::string marker, json config);

    void apply_direct_actions(const FlowStates<T>& fs, const GridBlock<MemModel, T>& grid,
                              Vector3s<T> vertex_vel);

    void apply_interp_actions(const GridBlock<MemModel, T>& grid, Vector3s<T> vertex_vel);

    void apply_constraints(const GridBlock<MemModel, T>& grid, Vector3s<T> vertex_vel);

private:
    std::vector<
        std::pair<std::string, std::shared_ptr<ShockFittingDirectVelocityAction<T, MemModel>>>>
        direct_actions_;
    std::vector<ShockFittingInterpolationAction<T, MemModel>> interp_actions_;
    std::vector<std::pair<std::string, std::shared_ptr<Constraint<T, MemModel>>>> constraints_;
};

template <typename T, class MemModel>
std::shared_ptr<ShockFittingDirectVelocityAction<T, MemModel>> make_direct_velocity_action(
    const GridBlock<MemModel, T>& grid, std::string marker, json config);

template <typename T, class MemModel>
std::shared_ptr<Constraint<T, MemModel>> make_constraint(json config);

template <typename T, class MemModel>
class ShockFitting : public GridMotionDriver<T, MemModel> {
public:
    ~ShockFitting() {}

    ShockFitting() {}

    ShockFitting(const GridBlock<MemModel, T>& grid, json config);

    void compute_vertex_velocities(const FlowStates<T>& fs, const GridBlock<MemModel, T>& grid,
                                   Vector3s<T> vertex_vel);

    void compute_boundary_velocities(const FlowStates<T>& fs, const GridBlock<MemModel, T>& grid,
                                     Vector3s<T> vertex_vel);

    void interpolate_internal_velocities(const GridBlock<MemModel, T>& grid,
                                         Vector3s<T> vertex_vel);

private:
    std::vector<ShockFittingBC<T, MemModel>> bcs_;

    // Compute the remaining vertex velocities
    ShockFittingInterpolationAction<T, MemModel> interp_;

    // think about a way to save memory by not storing the internal vertices,
    // and just using the vertices not in the boundary verties
    Field<size_t> internal_vertices_;
};

#endif
