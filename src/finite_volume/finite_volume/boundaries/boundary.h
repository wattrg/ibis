#ifndef BOUNDARY_H
#define BOUNDARY_H

#include <gas/flow_state.h>
#include <grid/grid.h>

enum class BoundaryConditions { SupersonicInflow, SlipWall, SupersonicOutflow };

template <typename T>
class PreReconstruction {
public:
    virtual ~PreReconstruction() {}

    virtual void apply(FlowStates<T>& fs, const GridBlock<T>& grid,
                       const Field<size_t>& boundary_faces) = 0;
};

template <typename T>
class FlowStateCopy : public PreReconstruction<T> {
public:
    FlowStateCopy(FlowState<T> fs) : fs_(fs) {}

    FlowStateCopy(json flow_state);

    ~FlowStateCopy() {}

    void apply(FlowStates<T>& fs, const GridBlock<T>& grid,
               const Field<size_t>& boundary_faces);

private:
    FlowState<T> fs_;
};

template <typename T>
class InternalCopy : public PreReconstruction<T> {
public:
    ~InternalCopy() {}

    void apply(FlowStates<T>& fs, const GridBlock<T>& grid,
               const Field<size_t>& boundary_faces);
};

template <typename T>
class InternalCopyReflectNormal : public PreReconstruction<T> {
public:
    ~InternalCopyReflectNormal() {}

    void apply(FlowStates<T>& fs, const GridBlock<T>& grid,
               const Field<size_t>& boundary_faces);
};

template <typename T>
class BoundaryCondition {
public:
    BoundaryCondition(json config);

    void apply_pre_reconstruction(FlowStates<T>& fs, const GridBlock<T>& grid,
                                  const Field<size_t>& boundary_faces);

private:
    std::vector<std::shared_ptr<PreReconstruction<T>>> pre_reconstruction_;
};

#endif
