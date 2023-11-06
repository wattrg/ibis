#include "conserved_quantities.h"

template <typename T>
ConservedQuantities<T>::ConservedQuantities(unsigned int n, unsigned int dim)
    : cq_(Kokkos::View<T**>("ConservedQuantities", n, dim+2)), num_values_(n), dim_(dim)
{
    mass_idx_ = 0;
    momentum_idx_ = 1;
    energy_idx_ = momentum_idx_ + dim;
}

template <typename T>
void ConservedQuantities<T>::apply_time_derivative(const ConservedQuantities<T>& dudt, double dt) {
    Kokkos::parallel_for("CQ::update_cq", num_values_, KOKKOS_CLASS_LAMBDA(const int i){
        mass(i) += dudt.mass(i) * dt;
        momentum_x(i) += dudt.momentum_x(i) * dt;
        momentum_y(i) += dudt.momentum_y(i) * dt;
        if (dim_ == 3){
            momentum_z(i) += dudt.momentum_z(i) * dt;
        }
        energy(i) += dudt.energy(i) * dt;
    });
}

template class ConservedQuantities<double>;
