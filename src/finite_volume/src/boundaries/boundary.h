#ifndef BOUNDARY_H
#define BOUNDARY_H

#include "../../../grid/src/grid.h"
#include "../../../gas/src/flow_state.h"

enum class BoundaryConditions {
    SupersonicInflow, SlipWall, SupersonicOutflow
};


template <typename T>
class PreReconstruction {
public:
    virtual void apply(FlowStates<T>& fs, GridBlock<T>& grid, Field<int>& boundary_faces)=0;
};

template <typename T>
class FlowStateCopy : public PreReconstruction<T> {
public:
    FlowStateCopy(FlowState<T> fs) : fs_(fs) {}

    KOKKOS_INLINE_FUNCTION
    void apply(FlowStates<T>& fs, GridBlock<T>& grid, Field<int>& boundary_faces) {
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

template <typename T>
class BoundaryCondition {
public:
    BoundaryCondition(std::vector<std::shared_ptr<PreReconstruction<T>>>);

    BoundaryCondition(json config);

    void apply_pre_reconstruction(FlowStates<T>& fs, GridBlock<T>& grid, Field<int>& boundary_faces);

private:
    std::vector<std::shared_ptr<PreReconstruction<T>>> pre_reconstruction_;
};

#endif
