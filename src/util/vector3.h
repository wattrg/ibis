#ifndef VECTOR3_H
#define VECTOR3_H

#include <Kokkos_Core.hpp>
#include "field.h"

namespace Aeolus {

typedef Kokkos::View<double*[3]> Vector3;

void dot(Vector3 &a, Vector3 &b, Field &result);
void add(Vector3 &a, Vector3 &b, Vector3 &result);
void subtract(Vector3 &a, Vector3 &b, Vector3 &result);
void cross(Vector3 &a, Vector3 &b, Vector3 &result);
void scale_in_place(Vector3 &a, double factor);
void length(Vector3 &a, Field &len);
void normalise(Vector3 &a);

}

#endif
