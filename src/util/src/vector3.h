#ifndef VECTOR3_H
#define VECTOR3_H

#include <Kokkos_Core.hpp>
#include <cmath>
#include "field.h"

namespace Aeolus {

// A collection of vectors with 3 components
typedef Kokkos::View<double*[3]> Vector3s;

void dot(Vector3s &a, Vector3s &b, Field &result);
void add(Vector3s &a, Vector3s &b, Vector3s &result);
void subtract(Vector3s &a, Vector3s &b, Vector3s &result);
void cross(Vector3s &a, Vector3s &b, Vector3s &result);
void scale_in_place(Vector3s &a, double factor);
void length(Vector3s &a, Field &len);
void normalise(Vector3s &a);

// A single vector with 3 components
struct Vector3 {
    Vector3(double x, double y, double z) : x(x), y(y), z(z) {}
    double x, y, z;

    bool operator == (const Vector3 &other) const {
        return (std::fabs(x - other.x) < 1e-14) &&
               (std::fabs(y - other.y) < 1e-14) &&
               (std::fabs(z - other.z) < 1e-14);
    }
};

}

#endif
