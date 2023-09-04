#ifndef CONSERVED_QUANTITIES_H
#define CONSERVED_QUANTITIES_H

#include "Kokkos_Macros.hpp"
#include <Kokkos_Core.hpp>

template <typename T>
class ConservedQuantities {
public:
    ConservedQuantities(unsigned int n, unsigned int dim)
        : cq_(Kokkos::View<T**>("ConservedQuantities", n, dim+2))
    {
        mass_idx_ = 0;
        momentum_idx_ = 1;
        energy_idx_ = momentum_idx_ + dim;
    }

    unsigned int size() const {return cq_.extent(0);}
    unsigned int n_conserved() const {return cq_.extent(1);}

    KOKKOS_FORCEINLINE_FUNCTION
    T& mass(int cell_i) const {return cq_(cell_i, mass_idx_);}
    
    KOKKOS_FORCEINLINE_FUNCTION
    T& mass(int cell_i) {return cq_(cell_i, mass_idx_);}

    KOKKOS_FORCEINLINE_FUNCTION
    T& momentum_x(int cell_i) const {return cq_(cell_i, momentum_idx_+0);}

    KOKKOS_FORCEINLINE_FUNCTION
    T& momentum_x(int cell_i) {return cq_(cell_i, momentum_idx_+0);}

    KOKKOS_FORCEINLINE_FUNCTION
    T& momentum_y(int cell_i) const {return cq_(cell_i, momentum_idx_+1);}

    KOKKOS_FORCEINLINE_FUNCTION 
    T& momentum_y(int cell_i) {return cq_(cell_i, momentum_idx_+1);}

    KOKKOS_FORCEINLINE_FUNCTION 
    T& momentum_z(int cell_i) const {return cq_(cell_i, momentum_idx_+2);}

    KOKKOS_FORCEINLINE_FUNCTION 
    T& momentum_z(int cell_i) {return cq_(cell_i, momentum_idx_+2);}

    KOKKOS_FORCEINLINE_FUNCTION
    T& energy(int cell_i) const {return cq_(cell_i, energy_idx_);}

    KOKKOS_FORCEINLINE_FUNCTION
    T& energy(int cell_i) {return cq_(cell_i, energy_idx_);}

private:
    Kokkos::View<T**> cq_;
    unsigned int mass_idx_, momentum_idx_, energy_idx_;
};

#endif
