#ifndef CONSERVED_QUANTITIES_H
#define CONSERVED_QUANTITIES_H

#include "Kokkos_Macros.hpp"
#include "../../gas/src/flow_state.h"
#include <Kokkos_Core.hpp>

template <typename T>
class ConservedQuantities {
public:
    ConservedQuantities(){}

    ConservedQuantities(unsigned int n, unsigned int dim);

    unsigned int size() const {return cq_.extent(0);}
    unsigned int n_conserved() const {return cq_.extent(1);}
    int dim() const {return dim_;}

    void apply_time_derivative(const ConservedQuantities<T>& dudt, double dt);

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
    int num_values_, dim_;
};

#endif
