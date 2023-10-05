#ifndef HANEL_H
#define HANEL_H

#include <Kokkos_Core.hpp>
#include "../../../gas/src/flow_state.h"
#include "../flux_calc.h"
#include "../conserved_quantities.h"

template <typename T>
void hanel(FlowStates<T>& left, FlowStates<T>& right, ConservedQuantities<T>& flux, bool three_d){
    Kokkos::parallel_for("Flux::hanel", flux.size(), KOKKOS_LAMBDA(const int i){
        // unpack left gas state
        T rL = left.gas.rho(i);
        T pL = left.gas.pressure(i);
        T eL = left.gas.energy(i);
        T aL = Kokkos::sqrt(1.4 * 287.0 * left.gas.temp(i));
        T uL = left.vel.x(i);
        T vL = left.vel.y(i);
        T wL = left.vel.z(i);
        T keL = 0.5*(uL*uL + vL*vL + wL*wL);
        T pLrL = pL / rL;
        T HL = eL + pLrL + keL;

        // unpack right gas state
        T rR = right.gas.rho(i);
        T pR = right.gas.pressure(i);
        T eR = right.gas.energy(i);
        T aR = Kokkos::sqrt(1.4 * 287.0 * right.gas.temp(i));
        T uR = right.vel.x(i);
        T vR = right.vel.y(i);
        T wR = right.vel.z(i);
        T keR = 0.5*(uR*uR + vR*vR + wR*wR);
        T pRrR = pR / rR;
        T HR = eR + pRrR + keR;


        // pressure and velocity splitting (eqn. 7 and 9)
        T pLplus, uLplus;
        if (Kokkos::abs(uL) < aL) {
            uLplus = 1.0 / (4.0*aL) * (uL+aL)*(uL+aL);
            pLplus = pL*uLplus * (1.0/aL * (2.0-uL/aL));
        }
        else {
            uLplus = 0.5*(uL+Kokkos::abs(uL));
            pLplus = pL*uLplus * (1.0/uL);
        }

        T pRminus, uRminus;
        if (Kokkos::abs(uR) < aR) {
            uRminus = -1.0/(4.0*aR) * (uR-aR)*(uR-aR);
            pRminus = pR*uRminus * (1.0/aR * (-2.0-uR/aR));
        }
        else {
            uRminus = 0.5*(uR-Kokkos::abs(uR));
            pRminus = pR*uRminus * (1.0/uR);
        }

        // the final fluxes
        T p_half = pLplus + pRminus;
        flux.mass(i) = uLplus*rL + uRminus * rR;
        flux.momentum_x(i) = uLplus * rL * uL + uRminus *rR * uR + p_half;
        flux.momentum_y(i) = uLplus * rL * vL + uRminus *rR * vR;
        if (three_d) {
            flux.momentum_z(i) = uLplus * rL * wL + uRminus * rR * wR;
        } 
        flux.energy(i) = uLplus * rL * HL + uRminus * rR * HR;
        // if (i==1 || i==3){
        //     // printf("face %i: rL = %f, rR = %f\n", i, rL, rR);
        //     // printf("face %i: uL = %f, uR = %f\n", i, uL, uR);
        //     // printf("face %i: rL = %f, rR = %f\n", i, rL, rR);
        //     // printf("face %i: hL = %f, hR = %f\n", i, HL, HR);
        //     // printf("face %i: aL = %f, aR = %f\n", i, aL, aR);
        //     // printf("face %i: keL = %f, keR = %f\n", i, keL, keR);
        //     printf("face %i: uLplus = %f, uRminus = %f\n", i, uLplus, uRminus);
        //     printf("face %i: pLplus = %f, pRminus = %f\n", i, pLplus, pRminus);
        //     printf("face %i: p_half = %f\n", i, p_half);
        //     printf("face %i: px = %f\n", i, flux.momentum_x(i));
        //     printf("\n");
        // }
    });
}

template void hanel<double>(FlowStates<double>&, FlowStates<double>&, ConservedQuantities<double>&, bool);

#endif
