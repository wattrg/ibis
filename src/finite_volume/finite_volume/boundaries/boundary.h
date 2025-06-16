#ifndef BOUNDARY_H
#define BOUNDARY_H

#include <gas/flow_state.h>
#include <gas/transport_properties.h>
#include <grid/grid.h>
#include <util/cubic_spline.h>

// #include "finite_volume/conserved_quantities.h"
#include <util/conserved_quantities.h>

enum class BoundaryConditions { SupersonicInflow, SlipWall, SupersonicOutflow };

template <typename T>
class GhostCellAction {
public:
    virtual ~GhostCellAction() {}

    virtual void apply(FlowStates<T>& fs, const GridBlock<T>& grid,
                       const Field<size_t>& boundary_faces, const IdealGas<T>& gas_model,
                       const TransportProperties<T>& trans_prop) = 0;
};

template <typename T>
class FlowStateCopy : public GhostCellAction<T> {
public:
    FlowStateCopy(FlowState<T> fs) : fs_(fs) {}

    FlowStateCopy(json flow_state);

    ~FlowStateCopy() {}

    void apply(FlowStates<T>& fs, const GridBlock<T>& grid,
               const Field<size_t>& boundary_faces, const IdealGas<T>& gas_model,
               const TransportProperties<T>& trans_prop);

private:
    FlowState<T> fs_;
};

template <typename T>
class BoundaryLayerProfile : public GhostCellAction<T> {
public:
    BoundaryLayerProfile(json config);

    ~BoundaryLayerProfile() {}

    void apply(FlowStates<T>& fs, const GridBlock<T>& grid,
               const Field<size_t>& boundary_faces, const IdealGas<T>& gas_model,
               const TransportProperties<T>& trans_prop);

private:
    CubicSpline v_;
    CubicSpline T_;
    Ibis::real p_;
};

template <typename T>
class InternalCopy : public GhostCellAction<T> {
public:
    ~InternalCopy() {}

    void apply(FlowStates<T>& fs, const GridBlock<T>& grid,
               const Field<size_t>& boundary_faces, const IdealGas<T>& gas_model,
               const TransportProperties<T>& trans_prop);
};

template <typename T>
class InternalCopyReflectNormal : public GhostCellAction<T> {
public:
    ~InternalCopyReflectNormal() {}

    void apply(FlowStates<T>& fs, const GridBlock<T>& grid,
               const Field<size_t>& boundary_faces, const IdealGas<T>& gas_model,
               const TransportProperties<T>& trans_prop);
};

template <typename T>
class InternalVelCopyReflect : public GhostCellAction<T> {
public:
    ~InternalVelCopyReflect() {}

    void apply(FlowStates<T>& fs, const GridBlock<T>& grid,
               const Field<size_t>& boundary_faces, const IdealGas<T>& gas_model,
               const TransportProperties<T>& trans_prop);
};

template <typename T>
class FixTemperature : public GhostCellAction<T> {
public:
    ~FixTemperature() {}

    FixTemperature(T temperature) : Twall_(temperature) {}

    void apply(FlowStates<T>& gs, const GridBlock<T>& grid,
               const Field<size_t>& boundary_faces, const IdealGas<T>& gas_model,
               const TransportProperties<T>& trans_prop);

private:
    T Twall_;
};

template <typename T>
class SubsonicInflow : public GhostCellAction<T> {
public:
    ~SubsonicInflow() {}

    SubsonicInflow(const FlowState<T> flow_state) : inflow_state_(flow_state) {}

    SubsonicInflow(json flow_state);

    void apply(FlowStates<T>& fs, const GridBlock<T>& grid,
               const Field<size_t>& boundary_faces, const IdealGas<T>& gas_model,
               const TransportProperties<T>& trans_prop);

private:
    FlowState<T> inflow_state_;
};

template <typename T>
class SubsonicOutflow : public GhostCellAction<T> {
public:
    ~SubsonicOutflow() {}

    SubsonicOutflow(T pressure) : pressure_(pressure) {}

    void apply(FlowStates<T>& fs, const GridBlock<T>& grid,
               const Field<size_t>& boundary_faces, const IdealGas<T>& gas_model,
               const TransportProperties<T>& trans_prop);

private:
    T pressure_;
};

template <typename T>
class FluxAction {
public:
    virtual ~FluxAction() {}

    virtual void apply(ConservedQuantities<T>& flux, const FlowStates<T>& fs,
                       const GridBlock<T>& grid, const Field<size_t>& boundary_faces,
                       const IdealGas<T>& gas_model,
                       const TransportProperties<T>& trans_prop) = 0;
};

template <typename T>
class ConstantFlux : public FluxAction<T> {
public:
    ConstantFlux() {}

    ConstantFlux(FlowState<T>& fs) : fs_(fs) {}

    ConstantFlux(json config) : fs_(FlowState<T>(config.at("flow_state"))) {}

    void apply(ConservedQuantities<T>& flux, const FlowStates<T>& fs,
               const GridBlock<T>& grid, const Field<size_t>& boundary_faces,
               const IdealGas<T>& gas_model, const TransportProperties<T>& trans_prop);

private:
    FlowState<T> fs_;
};

template <typename T>
class BoundaryCondition {
public:
    BoundaryCondition(json config);

    void apply_pre_reconstruction(FlowStates<T>& fs, const GridBlock<T>& grid,
                                  const Field<size_t>& boundary_faces,
                                  const IdealGas<T>& gas_model,
                                  const TransportProperties<T>& trans_prop);

    void apply_post_convective_flux_actions(ConservedQuantities<T>& flux,
                                            const FlowStates<T>& fs,
                                            const GridBlock<T>& grid,
                                            const Field<size_t>& boundary_faces,
                                            const IdealGas<T>& gas_model,
                                            const TransportProperties<T>& trans_prop);

    void apply_pre_viscous_grad(FlowStates<T>& fs, const GridBlock<T>& grid,
                                const Field<size_t>& boundary_faces,
                                const IdealGas<T>& gas_model,
                                const TransportProperties<T>& trans_prop);

private:
    std::vector<std::shared_ptr<GhostCellAction<T>>> pre_reconstruction_;
    std::vector<std::shared_ptr<GhostCellAction<T>>> pre_viscous_grad_;
    std::vector<std::shared_ptr<FluxAction<T>>> post_convective_flux_actions_;
};

#endif
