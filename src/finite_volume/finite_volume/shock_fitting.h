#ifndef SHOCK_FITTING_H
#define SHOCK_FITTING_H

#include <finite_volume/grid_motion_driver.h>

template <typename T>
class ShockFittingDirectVelocityAction {
public:
    virtual ~ShockFittingDirectVelocityAction() {}
    
    virtual void apply(const FlowStates<T>& fs, const GridBlock<T>& grid,
                       Vector3s<T> vertex_vel, const Field<size_t>& boundary_vertices) = 0;  
};

template <typename T>
class WaveSpeed : public ShockFittingDirectVelocityAction<T> {
public:
    ~WaveSpeed() {}

    void apply(const FlowStates<T>& fs, const GridBlock<T>& grid,
               Vector3s<T> vertex_vel, const Field<size_t>& boundary_vertices);  
};

template <typename T>
class ZeroVelocity : public ShockFittingDirectVelocityAction<T> {
public:
    ~ZeroVelocity() {}

    void apply(const FlowStates<T>& fs, const GridBlock<T>& grid,
               Vector3s<T> vertex_vel, const Field<size_t>& boundary_vertices,
               const Field<size_t>& boundary_faces);  
};

template <typename T>
class ShockFittingInterpolationAction {
public:
    virtual ~ShockFittingInterpolationAction() {}

    virtual void apply(const GridBlock<T>& grid, Vector3s<T> vertex_vel,
                       std::vector<const Field<size_t>> interpolation_points,
                       std::vector<const Field<size_t>> sample_points);
};

template <typename T>
class ConstrainDirection {
public:
    ~ConstrainDirection() {}

    void apply(const GridBlock<T>& grid,
               Vector3s<T> vertex_vel, std::vector<const Field<size_t>>& boundary_vertices);  

private:
    Vector3<T> direction_;
};

// template <typename T>
// class ShockFittingBC {
// public:
//     ~ShockFittingBC() {}

//     ShockFittingBC(json config);

// private:
//     // these directly set the velocity of vertices on boundaries
//     std::vector<std::shared_ptr<ShockFittingBCAction<T>>> direct_boundary_actions_;

//     // these interpolate the velocity of vertices on boundary,
//     // then potentially modify them to meet some constraint
//     std::vector<std::shared_ptr<ShockFittingBCAction<T>>> interp_boundary_action_;
// };

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
    // std::vector<ShockFittingBC<T>> bcs_;
    std::vector<Field<size_t>> boundary_interfaces_;
    std::vector<Field<size_t>> boundary_vertices_;

    // think about a way to save memory by not storing the internal vertices,
    // and just using the vertices not in the boundary verties
    Field<size_t> internal_vertices_;
};

#endif
