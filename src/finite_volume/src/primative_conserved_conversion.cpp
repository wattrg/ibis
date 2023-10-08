#include "primative_conserved_conversion.h"
#include "conserved_quantities.h"

template <typename T>
int conserved_to_primatives(ConservedQuantities<T>& cq, FlowStates<T> & fs)
{

    Kokkos::parallel_for("FS::from_conserved_quantities", fs.gas.size(), KOKKOS_LAMBDA(const int i){
        T rho = cq.mass(i);
        T vx = cq.momentum_x(i) / rho;
        T vy = cq.momentum_y(i) / rho;
        T vz = 0.0;
        if (cq.dim() == 3) {
            vz = cq.momentum_z(i) / rho;
        }
        T ke = 0.5*(vx*vx + vy*vy + vz*vz);
        T u = cq.energy(i)/rho - ke;
        fs.gas.rho(i) = rho;
        fs.vel.x(i) = vx;
        fs.vel.y(i) = vy;
        if (cq.dim() == 3){
            fs.vel.z(i) = vz;
        }
        fs.gas.energy(i) = u;
        fs.gas.temp(i) = fs.gas.energy(i) / 0.7171;
        fs.gas.pressure(i) = rho * 287.0 * fs.gas.temp(i);
    });
    return 0;
}
template int conserved_to_primatives(ConservedQuantities<double>& cq, FlowStates<double>& fs);

template <typename T>
int primatives_to_conserved(ConservedQuantities<T>& cq, FlowStates<T>& fs)
{
    Kokkos::parallel_for("CQ::from_flow_state", fs.gas.size(), KOKKOS_LAMBDA(const int i){
        T vx = fs.vel.x(i);
        T vy = fs.vel.y(i);
        T vz = fs.vel.z(i);
        T rho = fs.gas.rho(i);
        cq.mass(i) = fs.gas.rho(i);
        cq.momentum_x(i) = rho * vx;
        cq.momentum_y(i) = rho * vy;
        if (cq.dim() == 3) {
            cq.momentum_z(i) = rho * vz;
        }
        T ke = vx*vx + vy*vy;
        if (cq.dim() == 3){
            ke += vz*vz;
        }
        cq.energy(i) = rho*(0.5*ke + 0.7171*fs.gas.temp(i));
    });
    return 0;
}
template int primatives_to_conserved(ConservedQuantities<double>& cq, FlowStates<double>& fs);
