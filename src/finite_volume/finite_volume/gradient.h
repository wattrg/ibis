#ifndef GRADIENT_H
#define GRADIENT_H

#include <grid/grid.h>
#include <util/ragged_array.h>

#include <Kokkos_Core.hpp>
#include "Kokkos_Core_fwd.hpp"

template <typename T>
struct Gradients {
    Kokkos::View<T **> gradients;
};

template <typename T, class ExecSpace = Kokkos::DefaultExecutionSpace,
class Layout = Kokkos::DefaultExecutionSpace::array_layout>
class WLSGradient {
public:
    WLSGradient(const GridBlock<T> &block) {
        int num_connections = block.cells().neighbour_cells().num_values();
        typename Ibis::RaggedArray<T, Layout, ExecSpace>::ArrayType w ("weights", num_connections);
        auto offsets = block.cells().neighbour_cells().offsets(); 
        w_1_ = Ibis::RaggedArray<T, Layout, ExecSpace>(w, offsets);
        w_2_ = Ibis::RaggedArray<T, Layout, ExecSpace>(w, offsets);
        w_3_ = Ibis::RaggedArray<T, Layout, ExecSpace>(w, offsets);
        compute_workspace_(block);
    }

    template <class SubView>
    void compute_gradients(const SubView values, SubView gradients);

private:
    void compute_workspace_(const GridBlock<T> &block) {
        Kokkos::parallel_for(
            "WLSGradient::compute_workspace_::r", w_1_.num_rows(),
            KOKKOS_CLASS_LAMBDA (const int i) {
            auto neighbours = block.cells().neighbour_cells(i);
            T sum_dxdx = 0.0;
            T sum_dxdy = 0.0;
            T sum_dxdz = 0.0;
            T sum_dydy = 0.0;
            T sum_dydz = 0.0;
            T sum_dzdz = 0.0;
            T xi = block.cells().centroids().x(i);
            T yi = block.cells().centroids().y(i);
            T zi = block.cells().centroids().z(i);
            for (unsigned int j = 0; j < neighbours.size(); j++) {
                int neighbour_j = neighbours(j);
                T dx = block.cells().centroids().x(neighbour_j) - xi;
                T dy = block.cells().centroids().y(neighbour_j) - yi;
                T dz = block.cells().centroids().z(neighbour_j) - zi;
                sum_dxdx += dx*dx;
                sum_dxdy += dx*dy;
                sum_dxdz += dx*dz;
                sum_dydy += dy*dy;
                sum_dydz += dy*dz;
                sum_dzdz += dz*dz;
            }
            T r11 = Kokkos::sqrt(sum_dxdx);
            T r12 = 1.0 / r11 * sum_dxdy;
            T r22 = Kokkos::sqrt(sum_dydy - r12);
            T r13 = 1.0 / r11 * sum_dxdz;
            T r23 = 1.0 / r22 * sum_dydz - r12 / r11 * sum_dxdz;
            T r33 = Kokkos::sqrt(sum_dzdz - (r13*r13 + r23*r23));
            T beta = (r12*r23 - r13*r23) / (r11*r22);

            for (unsigned int j = 0; j < neighbours.size(); j++) {
                int neighbour_j = neighbours(j);
                T dx = block.cells().centroids().x(neighbour_j) - xi;
                T dy = block.cells().centroids().y(neighbour_j) - yi;
                T dz = block.cells().centroids().z(neighbour_j) - zi;
                T alpha_1 = dx / (r11 * r11);
                T alpha_2 = 1.0 / (r22*r22) * (dx - r12*r11*dx);
                T alpha_3 = 1.0 / (r33 * r33) * (dz - r23*r22*dy + beta*dx);
                w_1_(i, j) = alpha_1 - r12 / r11 * alpha_2 + beta * alpha_3;
                w_2_(i, j) = alpha_2 - r23 / r22 * alpha_3;
                w_3_(i, j) = alpha_3;
            }
        });
    }

private:
    Ibis::RaggedArray<T, Layout, ExecSpace> w_1_;
    Ibis::RaggedArray<T, Layout, ExecSpace> w_2_;
    Ibis::RaggedArray<T, Layout, ExecSpace> w_3_;
};

#endif
