#ifndef FINITE_VOLUME_H
#define FINITE_VOLUME_H

#include <nlohmann/json.hpp>

#include "../../gas/src/flow_state.h"
#include "../../grid/src/grid.h"
#include "boundaries/boundary.h"
#include "conserved_quantities.h"
#include "flux_calc.h"
#include "impl/Kokkos_HostThreadTeam.hpp"

using json = nlohmann::json;

template <typename T>
class FiniteVolume {
   public:
    FiniteVolume() {}

    FiniteVolume(const GridBlock<T>& grid, json config);

    int compute_dudt(FlowStates<T>& flow_state, const GridBlock<T>& grid,
                     ConservedQuantities<T>& dudt);

    double estimate_dt(const FlowStates<T>& flow_state, GridBlock<T>& grid);

    // methods
    // these have to be public for NVCC, but they shouldn't really need to be
    // accessed from outside of the class
    void apply_pre_reconstruction_bc(FlowStates<T>& fs,
                                     const GridBlock<T>& grid);
    void reconstruct(FlowStates<T>& flow_states, const GridBlock<T>& grid,
                     unsigned int order);
    void flux_surface_integral(const GridBlock<T>& grid,
                               ConservedQuantities<T>& dudt);
    void compute_flux(const GridBlock<T>& grid);
    void apply_post_convective_flux_bc();
    void apply_pre_spatial_deriv();
    int count_bad_cells(const FlowStates<T>& fs, const int num_cells);

   private:
    // memory
    FlowStates<T> left_;
    FlowStates<T> right_;
    ConservedQuantities<double> flux_;

    // boundary conditions
    std::vector<std::shared_ptr<BoundaryCondition<T>>> bcs_{};
    std::vector<Field<int>> bc_interfaces_{};

    // configuration
    unsigned int dim_;
    unsigned int reconstruction_order_;
    FluxCalculator flux_calculator_;
};

#endif
