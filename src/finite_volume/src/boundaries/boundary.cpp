#include "boundary.h"


template <typename T>
FlowStateCopy<T>::FlowStateCopy(json flow_state) {
    T temp = flow_state.at("T");
    T pressure = flow_state.at("p");

    GasState<T> gs;
    gs.temp = temp;
    gs.pressure = pressure;
    gs.rho = pressure / (287.0 * temp);
    gs.energy = 0.7171 * temp;

    T vx = flow_state.at("vx");
    T vy = flow_state.at("vy");
    T vz = flow_state.at("vz");
    Vector3<T> vel{vx, vy, vz};
    fs_ = FlowState<T>(gs, vel);
}

template <typename T>
void FlowStateCopy<T>::apply(FlowStates<T>& fs, const GridBlock<T>& grid, const Field<int>& boundary_faces) {
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
        fs.gas.temp(ghost_cell) = fs_.gas_state.temp;
        fs.gas.pressure(ghost_cell) = fs_.gas_state.pressure;
        fs.gas.rho(ghost_cell) = fs_.gas_state.rho;
        fs.gas.energy(ghost_cell) = fs_.gas_state.energy;

        fs.vel.x(ghost_cell) = fs_.velocity.x;
        fs.vel.y(ghost_cell) = fs_.velocity.y;
        fs.vel.z(ghost_cell) = fs_.velocity.z;
    }); 
}
template class FlowStateCopy<double>;


template <typename T>
BoundaryCondition<T>::BoundaryCondition(std::vector<std::shared_ptr<PreReconstruction<T>>> pre_reco)
    : pre_reconstruction_(pre_reco)
{}

template <typename T>
std::shared_ptr<PreReconstruction<T>> build_pre_reco(json config) {
    std::string type = config.at("type");
    std::shared_ptr<PreReconstruction<T>> action;
    if (type == "flow_state_copy"){
        action = std::shared_ptr<PreReconstruction<T>>(new FlowStateCopy<T>(config.at("flow_state")));
    }
    return action;
}

template <typename T>
BoundaryCondition<T>::BoundaryCondition(json config) {
    std::vector<json> pre_reco = config.at("pre_reconstruction");
    for (unsigned int i = 0; i < pre_reco.size(); i++){
        std::shared_ptr<PreReconstruction<T>> action = build_pre_reco<T>(pre_reco[i]);
        pre_reconstruction_.push_back(action);
    }
    
}

template <typename T>
void BoundaryCondition<T>::apply_pre_reconstruction(FlowStates<T>& fs, const GridBlock<T>& grid, const Field<int>& boundary_faces) {
    for (unsigned int i = 0; i < pre_reconstruction_.size(); i++){
        pre_reconstruction_[i]->apply(fs, grid, boundary_faces);
    }
}
template class BoundaryCondition<double>;
