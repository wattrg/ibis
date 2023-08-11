#ifndef VECTOR3_H
#define VECTOR3_H

#include <Kokkos_Core.hpp>
#include <cmath>
#include "field.h"

namespace Aeolus {

// A collection of vectors with 3 components
template <typename T>
using Vector3s = Kokkos::View<T*[3]>;

template <typename T>
void dot(Vector3s<T> &a, Vector3s<T> &b, Field<T> &result) {
    assert((a.extent(0) == b.extent(0)) && (b.extent(0) == result.extent(0)));
    
    Kokkos::parallel_for("vector3 dot product", a.extent(0), KOKKOS_LAMBDA(const int i){
        result(i) = a(i,0)*b(i,0) + a(i,1)*b(i,1) + a(i,2)*b(i,2);
    });
}

template <typename T>
void add(Vector3s<T> &a, Vector3s<T> &b, Vector3s<T> &result) {
    assert((a.extent(0) == b.extent(0)) && (b.extent(0) == result.extent(0)));

    Kokkos::parallel_for("Vector3s add", a.extent(0), KOKKOS_LAMBDA(const int i){
        result(i,0) = a(i,0) + b(i,0);
        result(i,1) = a(i,1) + b(i,1);
        result(i,2) = a(i,2) + b(i,2);
    });
}

template <typename T>
void subtract(Vector3s<T> &a, Vector3s<T> &b, Vector3s<T> &result) {
    assert((a.extent(0) == b.extent(0)) && (b.extent(0) == result.extent(0)));

    Kokkos::parallel_for("Vector3s subtract", a.extent(0), KOKKOS_LAMBDA(const int i) {
        result(i,0) = a(i,0) - b(i,0);
        result(i,1) = a(i,1) - b(i,1);
        result(i,2) = a(i,2) - b(i,2);
    });
}

template <typename T>
void cross(Vector3s<T> &a, Vector3s<T> &b, Vector3s<T> &result) {
    assert((a.extent(0) == b.extent(0)) && (b.extent(0) == result.extent(0)));

    Kokkos::parallel_for("Vector3s cross", a.extent(0), KOKKOS_LAMBDA(const int i) {
        result(i,0) = a(i,1)*b(i,2) - a(i,2)*b(i,1);
        result(i,1) = a(i,2)*b(i,0) - a(i,0)*b(i,2);
        result(i,2) = a(i,0)*b(i,1) - a(i,1)*b(i,0);
    });
}

template <typename T>
void scale_in_place(Vector3s<T> &a, T factor) {
    Kokkos::parallel_for("Vector3s scale", a.extent(0), KOKKOS_LAMBDA(const int i) {
        a(i, 0) *= factor;
        a(i, 1) *= factor;
        a(i, 2) *= factor;
    });
}

template <typename T>
void length(Vector3s<T> &a, Field<T> &len) {
    assert(a.extent(0) == len.extent(0));

    Kokkos::parallel_for("Vector3s length", a.extent(0), KOKKOS_LAMBDA(const int i) {
        len(i) = sqrt(a(i,0)*a(i,0) + a(i,1)*a(i,1) + a(i,2)*a(i,2));
    });
}

template <typename T>
void normalise(Vector3s<T> &a) {
    Kokkos::parallel_for("Vector3s normalise", a.extent(0), KOKKOS_LAMBDA(const int i) {
        double length_inv = 1./sqrt(a(i,0)*a(i,0) + a(i,1)*a(i,1) + a(i,2)*a(i,2));
        a(i,0) *= length_inv;
        a(i,1) *= length_inv;
        a(i,2) *= length_inv;
    });
}

// A single vector with 3 components
template <typename T>
struct Vector3 {
    Vector3(T x, T y, T z) : x(x), y(y), z(z) {}
    T x, y, z;

    bool operator == (const Vector3 &other) const {
        return (std::fabs(x - other.x) < 1e-14) &&
               (std::fabs(y - other.y) < 1e-14) &&
               (std::fabs(z - other.z) < 1e-14);
    }
};

}

#endif
