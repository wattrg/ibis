#include "boundary.h"

template <typename T>
BoundaryCondition<T>::BoundaryCondition(std::vector<std::shared_ptr<PreReconstruction<T>>> pre_reco)
    : pre_reconstruction_(pre_reco)
{}

template <typename T>
BoundaryCondition<T>::BoundaryCondition(json config) {
    (void) config;
}

template <typename T>
void BoundaryCondition<T>::apply_pre_reconstruction(FlowStates<T>& fs, GridBlock<T>& grid, Field<int>& boundary_faces) {
    for (unsigned int i = 0; i < pre_reconstruction_.size(); i++){
        pre_reconstruction_[i]->apply(fs, grid, boundary_faces);
    }
}


template class BoundaryCondition<double>;
