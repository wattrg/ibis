#include <finite_volume/shock_fitting.h>

template <typename T>
ShockFitting<T>::ShockFitting(json config) {
    (void)config;
}

template <typename T>
void ShockFitting<T>::compute_vertex_velocities(const FlowStates<T>& fs,
                                                const GridBlock<T>& grid,
                                                Vector3s<T> vertex_vel) {
  
    (void)fs;
    (void)grid;
    Kokkos::parallel_for(
        "shock_fitting", grid.num_vertices(), KOKKOS_LAMBDA(const size_t i) {
            vertex_vel.x(i) = T(1.0);
            vertex_vel.y(i) = T(0.0);
            vertex_vel.z(i) = T(0.0);
        });
}
template class ShockFitting<Ibis::real>;
template class ShockFitting<Ibis::dual>;
