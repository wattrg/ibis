#ifndef SHOCK_FITTING_H
#define SHOCK_FITTING_H

#include <finite_volume/grid_motion_driver.h>

template <typename T>
class ShockFittingDirectVelocityAction {
public:
    virtual ~ShockFittingDirectVelocityAction() {}

    virtual void apply(const FlowStates<T>& fs, const GridBlock<T>& grid,
                       Vector3s<T> vertex_vel,
                       const Field<size_t>& boundary_vertices) = 0;
};

template <typename T>
class WaveSpeed : public ShockFittingDirectVelocityAction<T> {
public:
    ~WaveSpeed() {}

    void apply(const FlowStates<T>& fs, const GridBlock<T>& grid, Vector3s<T> vertex_vel,
               const Field<size_t>& boundary_vertices);
};

template <typename T>
class ZeroVelocity : public ShockFittingDirectVelocityAction<T> {
public:
    ~ZeroVelocity() {}

    ZeroVelocity() {}

    void apply(const FlowStates<T>& fs, const GridBlock<T>& grid, Vector3s<T> vertex_vel,
               const Field<size_t>& boundary_vertices);
};

template <typename T>
class ShockFittingInterpolationAction {
public:
    ~ShockFittingInterpolationAction() {}

    ShockFittingInterpolationAction() {}

    ShockFittingInterpolationAction(Field<size_t> sample_points,
                                    Field<size_t> interp_points, Ibis::real power)
        : sample_points_(sample_points), interp_points_(interp_points), power_(power) {}

    ShockFittingInterpolationAction(const GridBlock<T>& grid,
                                    std::vector<std::string> sample_markers,
                                    std::string interp_marker, Ibis::real power);

    ShockFittingInterpolationAction(const GridBlock<T>& grid, std::string interp_marker,
                                    json config);

    void apply(const GridBlock<T>& grid, Vector3s<T> vertex_vel);

private:
    Field<size_t> sample_points_;
    Field<size_t> interp_points_;
    Ibis::real power_;
};

template <typename T>
class ConstrainDirection {
public:
    ~ConstrainDirection() {}

    ConstrainDirection(Ibis::real x, Ibis::real y, Ibis::real z)
        : direction_(Vector3<T>(x, y, z)) {}

    ConstrainDirection(json config);

    void apply(Vector3s<T> vertex_vel, Field<size_t>& boundary_vertices);

private:
    Vector3<T> direction_;
};

template <typename T>
class ShockFitting : public GridMotionDriver<T> {
public:
    ~ShockFitting() {}

    ShockFitting() {}

    ShockFitting(json config);

    void compute_vertex_velocities(const FlowStates<T>& fs, const GridBlock<T>& grid,
                                   Vector3s<T> vertex_vel);

    void compute_boundary_velocities(const FlowStates<T>& fs, const GridBlock<T>& grid,
                                     Vector3s<T> vertex_vel);

    void interpolate_internal_velocities(const GridBlock<T>& grid,
                                         Vector3s<T> vertex_vel);

private:
    // actions to perform for setting vertex velocities on boundaries
    // (the 'boundaries' for shock fitting may be intenral to the domain)
    std::vector<
        std::pair<std::string, std::shared_ptr<ShockFittingDirectVelocityAction<T>>>>
        direct_actions_;
    std::vector<ShockFittingInterpolationAction<T>> interp_actions_;
    std::vector<std::pair<std::string, ConstrainDirection<T>>> constraints_;

    // Compute the remaining vertex velocities
    ShockFittingInterpolationAction<T> final_interp_;

    // think about a way to save memory by not storing the internal vertices,
    // and just using the vertices not in the boundary verties
    Field<size_t> internal_vertices_;
};

#endif
