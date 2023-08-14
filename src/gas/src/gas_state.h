#ifndef GAS_H
#define GAS_H

#include <Kokkos_Core.hpp>

template <typename T>
struct GasStates {
public:
    GasStates(int n) {
        _rho_idx = 0;
        _pressure_idx = 1;
        _temp_idx = 2;
        _energy_idx = 3;
        _data = Kokkos::View<T**> ("GasStates", n, 4);
    }

    inline T& rho(const int cell_i) const {return _data(cell_i, _rho_idx);}
    inline T& rho(const int cell_i) {return _data(cell_i, _rho_idx);}

    inline T& pressure(const int cell_i) const {return _data(cell_i, _pressure_idx);}
    inline T& pressure(const int cell_i) {return _data(cell_i, _pressure_idx);}

    inline T& temp(const int cell_i) const {return _data(cell_i, _temp_idx);}
    inline T& temp(const int cell_i) {return _data(cell_i, _temp_idx);}

    inline T& energy(const int cell_i) const {return _data(cell_i, _energy_idx);}
    inline T& energy(const int cell_i) {return _data(cell_i, _energy_idx);}


private:
    Kokkos::View<T**> _data;
    int _rho_idx, _pressure_idx, _temp_idx, _energy_idx;
};

#endif
