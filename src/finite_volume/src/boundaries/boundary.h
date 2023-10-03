#ifndef BOUNDARY_H
#define BOUNDARY_H

#include "../../../grid/src/grid.h"
#include "../../../gas/src/flow_state.h"

enum class BoundaryConditions {
    SupersonicInflow, SlipWall, SupersonicOutflow
};

template <typename T>
class BoundaryCondition {
public:
    virtual void apply_pre_reconstruction(FlowStates<T>& fs, GridBlock<T>& grid, Field<int>& boundary_faces);
};

template <typename T>
class SupersonicInflow : public BoundaryCondition<T> {
public:
    SupersonicInflow(FlowState<T> fs) : fs_(fs) {}

    KOKKOS_INLINE_FUNCTION
    void apply_pre_reconstruction(FlowStates<T>& fs, GridBlock<T>& grid, Field<int>& boundary_faces) {
        unsigned int size = boundary_faces.size();
        Kokkos::parallel_for("SupersonicInflow::apply_pre_reconstruction", size, KOKKOS_LAMBDA(const int i){
            int face_id = boundary_faces(i);
            int left_cell = grid.interfaces().left_cell(face_id);
            int right_cell = grid.interfaces().right_cell(face_id);
            int ghost_cell;
            if (grid.is_valid(left_cell)) {
                ghost_cell = right_cell;
            }
            else {
                ghost_cell = left_cell;
            }
            fs.copy_flow_state(fs_, ghost_cell); 
        }); 
    }

private:
    FlowState<T> fs_;
};

#endif
