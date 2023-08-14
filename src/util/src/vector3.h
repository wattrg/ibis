#ifndef VECTOR3_H
#define VECTOR3_H

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include "field.h"

namespace Aeolus {

// A collection of vectors with 3 components
// template <typename T>
// using Vector3s = Kokkos::View<T*[3]>;

template <typename T>
struct Vector3s {
public:
    Vector3s() {}

    Vector3s(std::string description, int n) {
        _view = Kokkos::View<T*[3]>(description, n);
    }

    inline T& operator() (const int i, const int j) {
        return _view(i, j); 
    }

    inline T& operator() (const int i, const int j) const {
        return _view(i, j);
    }

    inline int size() const {return _view.extent(0);}

    inline bool operator == (const Vector3s &other) const {
        if (this->size() != other.size()) {
            return false;
        }

        for (int i = 0; i < this->size(); i++) {
            if (fabs(_view(i, 0) - other._view(i, 0)) > 1e-14)  return false;
            if (fabs(_view(i, 1) - other._view(i, 1)) > 1e-14)  return false;
            if (fabs(_view(i, 2) - other._view(i, 2)) > 1e-14)  return false;
        }
        return true;
    }

private:
    Kokkos::View<T*[3]> _view;
};

template <typename T>
void dot(Vector3s<T> &a, Vector3s<T> &b, Field<T> &result) {
    assert((a.size() == b.size()) && (b.size() == result.size()));
    
    Kokkos::parallel_for("vector3 dot product", a.size(), KOKKOS_LAMBDA(const int i){
        result(i) = a(i,0)*b(i,0) + a(i,1)*b(i,1) + a(i,2)*b(i,2);
    });
}

template <typename T>
void add(Vector3s<T> &a, Vector3s<T> &b, Vector3s<T> &result) {
    assert((a.size() == b.size()) && (b.size() == result.size()));

    Kokkos::parallel_for("Vector3s add", a.size(), KOKKOS_LAMBDA(const int i){
        result(i,0) = a(i,0) + b(i,0);
        result(i,1) = a(i,1) + b(i,1);
        result(i,2) = a(i,2) + b(i,2);
    });
}

template <typename T>
void subtract(Vector3s<T> &a, Vector3s<T> &b, Vector3s<T> &result) {
    assert((a.size() == b.size()) && (b.size() == result.size()));

    Kokkos::parallel_for("Vector3s subtract", a.size(), KOKKOS_LAMBDA(const int i) {
        result(i,0) = a(i,0) - b(i,0);
        result(i,1) = a(i,1) - b(i,1);
        result(i,2) = a(i,2) - b(i,2);
    });
}

template <typename T>
void cross(Vector3s<T> &a, Vector3s<T> &b, Vector3s<T> &result) {
    assert((a.size() == b.size()) && (b.size() == result.size()));

    Kokkos::parallel_for("Vector3s cross", a.size(), KOKKOS_LAMBDA(const int i) {
        result(i,0) = a(i,1)*b(i,2) - a(i,2)*b(i,1);
        result(i,1) = a(i,2)*b(i,0) - a(i,0)*b(i,2);
        result(i,2) = a(i,0)*b(i,1) - a(i,1)*b(i,0);
    });
}

template <typename T>
void scale_in_place(Vector3s<T> &a, T factor) {
    Kokkos::parallel_for("Vector3s scale", a.size(), KOKKOS_LAMBDA(const int i) {
        a(i, 0) *= factor;
        a(i, 1) *= factor;
        a(i, 2) *= factor;
    });
}

template <typename T>
void length(Vector3s<T> &a, Field<T> &len) {
    assert(a.size() == len.size());

    Kokkos::parallel_for("Vector3s length", a.size(), KOKKOS_LAMBDA(const int i) {
        len(i) = sqrt(a(i,0)*a(i,0) + a(i,1)*a(i,1) + a(i,2)*a(i,2));
    });
}

template <typename T>
void normalise(Vector3s<T> &a) {
    Kokkos::parallel_for("Vector3s normalise", a.size(), KOKKOS_LAMBDA(const int i) {
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
