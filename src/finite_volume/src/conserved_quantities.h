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

    T& mass(int cell_i) const {return _cq(cell_i, _mass);}
    T& mass(int cell_i) {return _cq(cell_i, _mass);}

    T& momentum(int cell_i, int direction) const {
        return _cq(cell_i, _momentum+direction);
    }
    T& momentum(int cell_i, int direction) {
        return _cq(cell_i, _momentum+direction);
    }

    T& energy(int cell_i) const {return _cq(cell_i, _energy);}
    T& energy(int cell_i) {return _cq(cell_i, _energy);}

private:
    Kokkos::View<T**> _cq;
    unsigned int _mass, _momentum, _energy;
};

#endif
