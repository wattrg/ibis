#include <finite_volume/flux_calc.h>
#include <gas/flow_state.h>
#include <util/numeric_types.h>

#include <Kokkos_Core.hpp>

template <typename T>
void Rusanov<T>::compute_flux(const FlowStates<T>& left, const FlowStates<T>& right,
                              ConservedQuantities<T>& flux, IdealGas<T>& gm,
                              bool three_d) {
    Kokkos::parallel_for(
        "rusanov", flux.size(), KOKKOS_LAMBDA(const int i) {
            T rL = left.gas.rho(i);
            T eL = gm.internal_energy(left.gas, i);
            T pL = left.gas.pressure(i);
            T uL = left.vel.x(i);
            T vL = left.vel.y(i);
            T wL = left.vel.z(i);
            T ruL = rL * uL;
            T keL = 0.5 * (uL * uL + vL * vL + wL * wL);
            T HL = eL + pL / rL + keL;
            T aL = gm.speed_of_sound(left.gas, i);

            T rR = right.gas.rho(i);
            T eR = gm.internal_energy(right.gas, i);
            T pR = right.gas.pressure(i);
            T uR = right.vel.x(i);
            T vR = right.vel.y(i);
            T wR = right.vel.z(i);
            T ruR = rR * uR;
            T keR = 0.5 * (uR * uR + vR * vR + wR * wR);
            T HR = eR + pR / rR + keR;
            T aR = gm.speed_of_sound(right.gas, i);

            T S_plus = Ibis::max(Ibis::abs(uL - aL), Ibis::abs(uR - aR));
            S_plus = Ibis::max(S_plus, Ibis::abs(uL + aL));
            S_plus = Ibis::max(S_plus, Ibis::abs(uR + aR));

            flux.mass(i) = 0.5 * (ruL + ruR) - 0.5 * S_plus * (rR - rL);
            flux.momentum_x(i) = 0.5 * (ruL * uL + pL + ruR * uR + pR) -
                                 0.5 * S_plus * (rR * uR - rL * uL);
            flux.momentum_y(i) =
                0.5 * (ruL * vL + ruR * vR) - 0.5 * S_plus * (rR * vR - rL * vL);
            if (three_d) {
                flux.momentum_z(i) =
                    0.5 * (ruL * wL + ruR * wR) - 0.5 * S_plus * (rR * wR - rL * wL);
            }
            flux.energy(i) =
                0.5 * (ruL * HL + ruR * HR) - 0.5 * S_plus * (rR * HR - rL * HL);
        });
}
template class Rusanov<Ibis::real>;
template class Rusanov<Ibis::dual>;
