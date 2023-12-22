#ifndef GRADIENT_H
#define GRADIENT_H

#include <Kokkos_Core.hpp>
#include <util/ragged_array.h>
#include <grid/grid.h>

template <typename T>
struct Gradients{
    Kokkos::View<T**> gradients;
};

template <typename T>
class WLSGradient{
public:
    WLSGradient(const GridBlock<T> &block);

    template <class SubView>
    void compute_gradients(const SubView values, SubView gradients);

private:
    void compute_workspace_(const GridBlock<T> &block);

private:
    Ibis::RaggedArray<T> w_;
};

#endif
