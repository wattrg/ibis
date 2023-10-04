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

    virtual ~PreReconstruction(){}

    virtual void apply(FlowStates<T>& fs, const GridBlock<T>& grid, const Field<int>& boundary_faces) = 0;
};

template <typename T>
class FlowStateCopy : public PreReconstruction<T> {
public:
    FlowStateCopy(FlowState<T> fs) : fs_(fs) {}

    FlowStateCopy(json flow_state);

    ~FlowStateCopy(){}

    void apply(FlowStates<T>& fs, const GridBlock<T>& grid, const Field<int>& boundary_faces);

private:
    FlowState<T> fs_;
};

template <typename T>
class BoundaryCondition {
public:
    BoundaryCondition(std::vector<std::shared_ptr<PreReconstruction<T>>>);

    BoundaryCondition(json config);

    void apply_pre_reconstruction(FlowStates<T>& fs, const GridBlock<T>& grid, const Field<int>& boundary_faces);

private:
    std::vector<std::shared_ptr<PreReconstruction<T>>> pre_reconstruction_;
};

#endif
