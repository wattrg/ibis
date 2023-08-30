#ifndef CONSERVED_QUANTITIES_H
#define CONSERVED_QUANTITIES_H

#include <Kokkos_Core.hpp>

template <typename T>
struct ConservedQuantities {
public:
    ConservedQuantities(unsigned int n, unsigned int dim)
        : _cq(Kokkos::View<T**>("ConservedQuantities", n, dim+2))
    {
        _mass = 0;
        _momentum = 1;
        _energy = _momentum + dim;
    }

    unsigned int size() const {return _cq.extent(0);}
    unsigned int n_conserved() const {return _cq.extent(1);}

    inline T& mass(int cell_i) const {return _cq(cell_i, _mass);}
    inline T& mass(int cell_i) {return _cq(cell_i, _mass);}

    inline T& momentum_x(int cell_i) const {return _cq(cell_i, _momentum+0);}
    inline T& momentum_x(int cell_i) {return _cq(cell_i, _momentum+0);}

    inline T& momentum_y(int cell_i) const {return _cq(cell_i, _momentum+1);}
    inline T& momentum_y(int cell_i) {return _cq(cell_i, _momentum+1);}

    inline T& momentum_z(int cell_i) const {return _cq(cell_i, _momentum+2);}
    inline T& momentum_z(int cell_i) {return _cq(cell_i, _momentum+2);}

    inline T& energy(int cell_i) const {return _cq(cell_i, _energy);}
    inline T& energy(int cell_i) {return _cq(cell_i, _energy);}

private:
    Kokkos::View<T**> _cq;
    unsigned int _mass, _momentum, _energy;
};

#endif
