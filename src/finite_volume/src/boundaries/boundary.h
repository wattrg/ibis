#ifndef BOUNDARY_H
#define BOUNDARY_H

#include "../../../gas/src/flow_state.h"

enum class BoundaryConditions {
    SupersonicInflow, SlipWall, SupersonicOutflow
};

template <typename T>
class SupersonicInflow {
public:
    SupersonicInflow(FlowState<T> fs) : fs_(fs) {}

    KOKKOS_INLINE_FUNCTION
    void apply_pre_reconstruction(FlowStates<T>& ghost) {
        Kokkos::parallel_for("SupersonicInflow::apply_pre_reconstruction", ghost.size(), KOKKOS_LAMBDA(const int i){
            ghost.copy_flow_state(fs_, i); 
        }); 
    }

private:
    FlowState<T> fs_;
};

#endif
