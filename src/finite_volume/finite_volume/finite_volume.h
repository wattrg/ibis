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

template <typename T>
class FiniteVolume {
public:
    FiniteVolume() {}

    FiniteVolume(const GridBlock<T>& grid, json config);

    size_t compute_dudt(FlowStates<T>& flow_state, const GridBlock<T>& grid,
                        ConservedQuantities<T>& dudt, IdealGas<T>& gas_model,
                        TransportProperties<T>& trans_prop);

    double estimate_dt(const FlowStates<T>& flow_state, GridBlock<T>& grid,
                       IdealGas<T>& gas_model,
                       TransportProperties<T>& trans_prop);

    // methods
    // these have to be public for NVCC, but they shouldn't really need to
    // be accessed from outside of the class. Although sometimes the
    // post-processing will call them.
    void apply_pre_reconstruction_bc(FlowStates<T>& fs,
                                     const GridBlock<T>& grid);
    void apply_pre_viscous_grad_bc(FlowStates<T>& fs, const GridBlock<T>& grid);
    void reconstruct(FlowStates<T>& flow_states, const GridBlock<T>& grid,
                     IdealGas<T>& gas_model, TransportProperties<T>& trans_prop,
                     size_t order);
    void copy_reconstruct(FlowStates<T>& flow_states, const GridBlock<T>& grid);
    void linear_reconstruct(const FlowStates<T>& flow_states,
                            const GridBlock<T>& grid, IdealGas<T>& gas_model,
                            TransportProperties<T>& trans_prop);
    void flux_surface_integral(const GridBlock<T>& grid,
                               ConservedQuantities<T>& dudt);
    void compute_convective_flux(const GridBlock<T>& grid,
                                 IdealGas<T>& gas_model);
    void compute_viscous_flux(const FlowStates<T>& flow_states,
                              const GridBlock<T>& grid,
                              const IdealGas<T>& gas_model,
                              const TransportProperties<T>& trans_prop);
    void compute_viscous_properties_at_faces(const FlowStates<T>& flow_states,
                                             const GridBlock<T>& grid,
                                             const IdealGas<T>& gas_model);
    size_t count_bad_cells(const FlowStates<T>& fs, const size_t num_cells);

public:
    // methods for IO
    const Gradients<T>& cell_gradients() const { return cell_grad_; }

private:
    // memory
    FlowStates<T> left_;
    FlowStates<T> right_;
    ConservedQuantities<T> flux_;

    // boundary conditions
    std::vector<std::shared_ptr<BoundaryCondition<T>>> bcs_{};
    std::vector<Field<size_t>> bc_interfaces_{};

    // configuration
    size_t dim_;
    size_t reconstruction_order_;
    FluxCalculator flux_calculator_;
    bool viscous_;

    // gradients
    WLSGradient<T> grad_calc_;
    Gradients<T> cell_grad_;

    // memory for viscous fluxes
    FlowStates<T> face_fs_;
    Gradients<T> face_grad_;

    // limiter
    Limiter<T> limiter_;
    LimiterValues<T> limiters_;
};

#endif
