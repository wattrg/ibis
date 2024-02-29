#ifndef FINITE_VOLUME_H
#define FINITE_VOLUME_H

#include <finite_volume/boundaries/boundary.h>
#include <finite_volume/conserved_quantities.h>
#include <finite_volume/flux_calc.h>
#include <finite_volume/gradient.h>
#include <finite_volume/limiter.h>
#include <gas/flow_state.h>
#include <gas/gas_model.h>
#include <gas/transport_properties.h>
#include <grid/grid.h>
#include <spdlog/spdlog.h>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

/**
 * Handles the fluid dynamics/finite volume related things
 *
 * @tparam T The type of number flow variables are stored with
 */
template <typename T>
class FiniteVolume {
public:
    FiniteVolume() {}

    FiniteVolume(const GridBlock<T>& grid, json config);

    /**
     * Compute the time derivative of a particular flow state
     *
     * @param[in] flow_state The flow state to compute the time derivative for
     * @param[in] grid The grid
     * @param[out] dudt The time derivatives of conserved quantities
     * @param[in] gas_model The gas model
     * @param[in] trans_prop The transport properties
     */
    size_t compute_dudt(FlowStates<T>& flow_state, const GridBlock<T>& grid,
                        ConservedQuantities<T>& dudt, IdealGas<T>& gas_model,
                        TransportProperties<T>& trans_prop);

    /**
     * Estimate the allowable global time step for a given flow
     * state on a given grid
     *
     * @param flow_state The flow state to estimate the allowable
     *     time step for
     * @param grid The grid to compute the time step for
     * @param gas_model The gas model
     * @param trans_prop The transport properties
     * @return the size of the time step
     */
    double estimate_dt(const FlowStates<T>& flow_state, GridBlock<T>& grid,
                       IdealGas<T>& gas_model,
                       TransportProperties<T>& trans_prop);

    // methods
    // these have to be public for NVCC, but they shouldn't really need to
    // be accessed from outside of the class. Although sometimes the
    // post-processing will call them.

    // Apply pre-reconstruction boundary conditions
    void apply_pre_reconstruction_bc(FlowStates<T>& fs,
                                     const GridBlock<T>& grid);

    // Apply pre-reconstruction boundary conditions
    void apply_pre_viscous_grad_bc(FlowStates<T>& fs, const GridBlock<T>& grid);

    // Perform reconstruction
    void reconstruct(FlowStates<T>& flow_states, const GridBlock<T>& grid,
                     IdealGas<T>& gas_model, TransportProperties<T>& trans_prop,
                     size_t order);

    // Apply copy-reconstruction
    void copy_reconstruct(FlowStates<T>& flow_states, const GridBlock<T>& grid);

    // Perform linear reconstruction
    void linear_reconstruct(const FlowStates<T>& flow_states,
                            const GridBlock<T>& grid, IdealGas<T>& gas_model,
                            TransportProperties<T>& trans_prop);

    // Perform the surface integral of fluxes over the cells
    void flux_surface_integral(const GridBlock<T>& grid,
                               ConservedQuantities<T>& dudt);

    // Compute the convective fluxes
    void compute_convective_flux(const GridBlock<T>& grid,
                                 IdealGas<T>& gas_model);

    // Compute the viscous fluxes
    void compute_viscous_flux(const FlowStates<T>& flow_states,
                              const GridBlock<T>& grid,
                              const IdealGas<T>& gas_model,
                              const TransportProperties<T>& trans_prop);

    // Compute viscous properties at faces
    void compute_viscous_properties_at_faces(const FlowStates<T>& flow_states,
                                             const GridBlock<T>& grid,
                                             const IdealGas<T>& gas_model);

    // Count the number of bad cells in the domain
    size_t count_bad_cells(const FlowStates<T>& fs, const size_t num_cells);

public:
    // methods for IO
    const Gradients<T>& cell_gradients() const { return cell_grad_; }

private:
    // The flow states to the left of the interfaces
    FlowStates<T> left_;

    // The flow states to the right of the interfaces
    FlowStates<T> right_;

    // The flux values
    ConservedQuantities<T> flux_;

    // boundary conditions
    std::vector<std::shared_ptr<BoundaryCondition<T>>> bcs_{};

    // The interfaces on each boundary
    std::vector<Field<size_t>> bc_interfaces_{};

    // number of spatial dimensions
    size_t dim_;

    // The reconstruction order
    size_t reconstruction_order_;

    // The flux calculator
    std::unique_ptr<FluxCalculator<T>> flux_calculator_;

    // Flag for whether viscous fluxes should be included
    bool viscous_;

    // Gradient calculator
    WLSGradient<T> grad_calc_;

    // Storage for gradients at cells
    Gradients<T> cell_grad_;

    // Flow states at interfaces, for viscous fluxes
    FlowStates<T> face_fs_;

    // Storage for gradients at interfaces
    Gradients<T> face_grad_;

    // The limiter
    Limiter<T> limiter_;

    // Storage for limiter values
    LimiterValues<T> limiters_;
};

#endif
