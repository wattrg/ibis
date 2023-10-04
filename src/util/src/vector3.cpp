#include <cmath>
#include <cassert>
#include "vector3.h"

#include <doctest/doctest.h>
#include <Kokkos_MathematicalFunctions.hpp>

#define VEC3_TOL 1e-15

template <typename T>
void dot(const Vector3s<T> &a, const Vector3s<T> &b, Field<T> &result) {
    assert((a.size() == b.size()) && (b.size() == result.size()));
    
    Kokkos::parallel_for("vector3 dot product", a.size(), KOKKOS_LAMBDA(const int i){
        result(i) = a(i,0)*b(i,0) + a(i,1)*b(i,1) + a(i,2)*b(i,2);
    });
}
template void dot<double>(const Vector3s<double> &a, const Vector3s<double> &b, Field<double> &result);

template <typename T>
void add(const Vector3s<T> &a, const Vector3s<T> &b, Vector3s<T> &result) {
    assert((a.size() == b.size()) && (b.size() == result.size()));

    Kokkos::parallel_for("Vector3s add", a.size(), KOKKOS_LAMBDA(const int i){
        result(i,0) = a(i,0) + b(i,0);
        result(i,1) = a(i,1) + b(i,1);
        result(i,2) = a(i,2) + b(i,2);
    });
}
template void add<double>(const Vector3s<double> &a, const Vector3s<double> &b, Vector3s<double> &result);

template <typename T>
void subtract(const Vector3s<T> &a, const Vector3s<T> &b, Vector3s<T> &result) {
    assert((a.size() == b.size()) && (b.size() == result.size()));

    Kokkos::parallel_for("Vector3s subtract", a.size(), KOKKOS_LAMBDA(const int i) {
        result(i,0) = a(i,0) - b(i,0);
        result(i,1) = a(i,1) - b(i,1);
        result(i,2) = a(i,2) - b(i,2);
    });
}
template void subtract<double>(const Vector3s<double> &a, const Vector3s<double> &b, Vector3s<double> &result);

template <typename T>
void cross(const Vector3s<T> &a, const Vector3s<T> &b, Vector3s<T> &result) {
    assert((a.size() == b.size()) && (b.size() == result.size()));

    Kokkos::parallel_for("Vector3s cross", a.size(), KOKKOS_LAMBDA(const int i) {
        result(i,0) = a(i,1)*b(i,2) - a(i,2)*b(i,1);
        result(i,1) = a(i,2)*b(i,0) - a(i,0)*b(i,2);
        result(i,2) = a(i,0)*b(i,1) - a(i,1)*b(i,0);
    });
}
template void cross<double>(const Vector3s<double> &a, const Vector3s<double> &b, Vector3s<double> &result);

template <typename T>
void scale_in_place(Vector3s<T> &a, T factor) {
    Kokkos::parallel_for("Vector3s scale", a.size(), KOKKOS_LAMBDA(const int i) {
        a(i, 0) *= factor;
        a(i, 1) *= factor;
        a(i, 2) *= factor;
    });
}
template void scale_in_place<double>(Vector3s<double> &a, double factor);


template <typename T>
void length(const Vector3s<T> &a, Field<T> &len) {
    assert(a.size() == len.size());

    Kokkos::parallel_for("Vector3s length", a.size(), KOKKOS_LAMBDA(const int i) {
        len(i) = sqrt(a(i,0)*a(i,0) + a(i,1)*a(i,1) + a(i,2)*a(i,2));
    });
}
template void length<double>(const Vector3s<double> &a, Field<double> &len);

template <typename T>
void normalise(Vector3s<T> &a) {
    Kokkos::parallel_for("Vector3s normalise", a.size(), KOKKOS_LAMBDA(const int i) {
        double length_inv = 1./sqrt(a(i,0)*a(i,0) + a(i,1)*a(i,1) + a(i,2)*a(i,2));
        a(i,0) *= length_inv;
        a(i,1) *= length_inv;
        a(i,2) *= length_inv;
    });
}
template void normalise<double>(Vector3s<double> &a);

template <typename T>
void transform_to_local_frame(Vector3s<T>& a, const Vector3s<T>& norm, 
                              const Vector3s<T>& tan1, const Vector3s<T> tan2)
{
    Kokkos::parallel_for("Vector3s::transform_to_local_frame", a.size(), KOKKOS_LAMBDA(const int i){
        T x = a.x(i) * norm.x(i) + a.y(i) * norm.y(i) + a.z(i) * norm.z(i);
        T y = a.x(i) * tan1.x(i) + a.y(i) * tan1.y(i) + a.z(i) * tan1.z(i);
        T z = a.x(i) * tan2.x(i) + a.y(i) * tan2.y(i) + a.z(i) * tan2.z(i);
        a.x(i) = x;
        a.y(i) = y;
        a.z(i) = z;
    });
}
template void transform_to_local_frame<double>(Vector3s<double>& a, const Vector3s<double>& norm, 
                              const Vector3s<double>& tan1, const Vector3s<double> tan2);

template <typename T>
void transform_to_global_frame(Vector3s<T>& a, const Vector3s<T>& norm, 
                               const Vector3s<T>& tan1, const Vector3s<T>& tan2)
{
    Kokkos::parallel_for("Vector3s::transform_to_global_frame", a.size(), KOKKOS_LAMBDA(const int i){
        T x = a(i, 0) * norm(i, 0) + a(i, 1) * tan1(i, 0) + a(i, 2) * tan2(i, 0);
        T y = a(i, 0) * norm(i, 1) + a(i, 1) * tan1(i, 1) + a(i, 2) * tan2(i, 1);
        T z = a(i, 0) * norm(i, 2) + a(i, 1) * tan1(i, 2) + a(i, 2) * tan2(i, 2);
        a.x(i) = x;
        a.y(i) = y;
        a.z(i) = z;
    });
}

template void transform_to_global_frame<double>(Vector3s<double>& a, const Vector3s<double>& norm, const Vector3s<double>& tan1, const Vector3s<double>& tan2);

TEST_CASE("Vector Dot Product") {
    int n = 10;
    Vector3s<double> a ("a", n);
    Vector3s<double> b ("b", n);
    Field<double> result ("result", n);
    Field<double> expected ("expected", n);
    
    // set some data
    for (int i = 0; i < n; i++) {
        a(i,0) = 1.0 * i;
        a(i,1) = 2.0 * i;
        a(i,2) = 3.0 * i;

        b(i,0) = 1.0 * i * i;
        b(i,1) = 2.0 * i * i;
        b(i,2) = 3.0 * i * i;

        expected(i) = a(i,0)*b(i,0) + a(i,1)*b(i,1) + a(i,2)*b(i,2);
    }

    dot(a, b, result);

    // check the results are correct
    for (int i = 0; i < n; i++) {
        CHECK(Kokkos::fabs(expected(i) < result(i)) < VEC3_TOL);
    }
}

TEST_CASE("Vector3s Add") {
    int n = 20;
    Vector3s<double> a ("a", n);
    Vector3s<double> b ("b", n);
    Vector3s<double> result ("result", n);
    Vector3s<double> expected ("expected", n);
    
    // set some data
    for (int i = 0; i < n; i++) {
        a(i,0) = 1.0 * i;
        a(i,1) = - 2.0 * i;
        a(i,2) = 3.0 * i - 5;

        b(i,0) = - 1.0 * i * i;
        b(i,1) = 0.5 * (i-1) * i;
        b(i,2) = 3.0 * i * i;

        expected(i,0) = a(i,0) + b(i,0); 
        expected(i,1) = a(i,1) + b(i,1);
        expected(i,2) = a(i,2) + b(i,2);
    }

    add(a, b, result);

    // check the results are correct
    for (int i = 0; i < n; i++) {
        CHECK(Kokkos::fabs(expected(i,0) - result(i,0)) < VEC3_TOL);
        CHECK(Kokkos::fabs(expected(i,1) - result(i,1)) < VEC3_TOL);
        CHECK(Kokkos::fabs(expected(i,2) - result(i,2)) < VEC3_TOL);
    }
}

TEST_CASE("Vector3s subtract") {
    int n = 20;
    Vector3s<double> a ("a", n);
    Vector3s<double> b ("b", n);
    Vector3s<double> result ("result", n);
    Vector3s<double> expected ("expected", n);
    
    // set some data
    for (int i = 0; i < n; i++) {
        a(i,0) = 1.0 * i;
        a(i,1) = - 2.0 * i;
        a(i,2) = 3.0 * i - 5;

        b(i,0) = - 1.0 * i * i;
        b(i,1) = 0.5 * (i-1) * i;
        b(i,2) = 3.0 * i * i;

        expected(i,0) = a(i,0) - b(i,0); 
        expected(i,1) = a(i,1) - b(i,1);
        expected(i,2) = a(i,2) - b(i,2);
    }

    subtract(a, b, result);

    // check the results are correct
    for (int i = 0; i < n; i++) {
        CHECK(Kokkos::fabs(expected(i,0) - result(i,0)) < VEC3_TOL);
        CHECK(Kokkos::fabs(expected(i,1) - result(i,1)) < VEC3_TOL);
        CHECK(Kokkos::fabs(expected(i,2) - result(i,2)) < VEC3_TOL);
    }
}

TEST_CASE("Vector3s cross") {
    int n = 20;
    Vector3s<double> a ("a", n);
    Vector3s<double> b ("b", n);
    Vector3s<double> result ("result", n);
    Vector3s<double> expected ("expected", n);
    
    // set some data
    for (int i = 0; i < n; i++) {
        a(i,0) = 1.0 * i;
        a(i,1) = - 2.0 * i;
        a(i,2) = 3.0 * i - 5;

        b(i,0) = - 1.0 * i * i;
        b(i,1) = 0.5 * (i-1) * i;
        b(i,2) = 3.0 * i * i;

        expected(i,0) = a(i,1)*b(i,2) - a(i,2)*b(i,1);
        expected(i,1) = a(i,2)*b(i,0) - a(i,0)*b(i,2);
        expected(i,2) = a(i,0)*b(i,1) - a(i,1)*b(i,0);
    }

    cross(a, b, result);

    // check the results are correct
    for (int i = 0; i < n; i++) {
        CHECK(Kokkos::fabs(expected(i,0) - result(i,0)) < VEC3_TOL);
        CHECK(Kokkos::fabs(expected(i,1) - result(i,1)) < VEC3_TOL);
        CHECK(Kokkos::fabs(expected(i,2) - result(i,2)) < VEC3_TOL);
    }
}

TEST_CASE("Vector3s scale_in_place") {
    int n = 20;
    Vector3s<double> a ("a", n);
    double factor = 2.0;

    for (int i = 0; i < n; i++){
        a(i, 0) = 1.0 * i;
        a(i, 1) = 2.0 * i;
        a(i, 2) = 3.0 * i;
    }

    scale_in_place(a, factor);

    for (int i = 0; i < n; i++){
        CHECK(Kokkos::fabs(a(i, 0) - 2.0 * i) < VEC3_TOL);
        CHECK(Kokkos::fabs(a(i, 1) - 4.0 * i) < VEC3_TOL);
        CHECK(Kokkos::fabs(a(i, 2) - 6.0 * i) < VEC3_TOL);
    }
}

TEST_CASE("Vector3s length") {
    int n = 20;
    Vector3s<double> a ("a", n);
    Field<double> len ("length", n);
    Field<double> expected ("expected", n);

    for (int i = 0; i < n; i++){
        a(i, 0) = 1.0 * i;
        a(i, 1) = 2.0 * i;
        a(i, 2) = 3.0 * i;

        expected(i) = sqrt(a(i,0)*a(i,0) + a(i,1)*a(i,1) + a(i,2)*a(i,2));
    }

    length(a, len);

    for (int i = 0; i < n; i++){
        CHECK(Kokkos::fabs(len(i) - expected(i)) < VEC3_TOL);
    }
}

TEST_CASE("Vector3s normalise") {
    int n = 20;
    Vector3s<double> a ("a", n);

    for (int i = 0; i < n; i++){
        a(i, 0) = 1.0 * i + 1.0;
        a(i, 1) = 2.0 * i;
        a(i, 2) = 3.0 * i;
    }

    normalise(a);

    for (int i = 0; i < n; i++){
        double length_inv = sqrt((i+1.0)*(i+1.0) + 4.0*i*i + 9.0*i*i);
        CHECK(Kokkos::fabs(a(i,0) - 1.0 * (i+1.0) / length_inv) < VEC3_TOL);
        CHECK(Kokkos::fabs(a(i,1) - 2.0 * i / length_inv) < VEC3_TOL);
        CHECK(Kokkos::fabs(a(i,2) - 3.0 * i / length_inv) < VEC3_TOL);
    }
}

TEST_CASE("Vector3s::transform_to_local_frame") {
    int n = 3;
    Vector3s<double> a ("a", n);
    Vector3s<double> norm ("norm", n);
    Vector3s<double> tan1 ("tan1", n);
    Vector3s<double> tan2 ("tan2", n);

    a.x(0) = 1.0; a.y(0) = 1.0;
    norm.x(0) = 1.0; norm.y(0) = 0.0;
    tan1.x(0) = 0.0; tan1.y(0) = 1.0;
    tan2.z(0) = 1.0;

    a.x(1) = 1.0; a.y(1) = 0.0;
    norm.x(1) = 0.0; norm.y(1) = 1.0;
    tan1.x(1) = 1.0; tan1.y(1) = 0.0;
    tan2.z(1) = 1.0;

    a.x(2) = 1.0; a.y(2) = 1.0;
    norm.x(2) = -1/Kokkos::sqrt(2); norm.y(2) = 1/Kokkos::sqrt(2);
    tan1.x(2) = 1/Kokkos::sqrt(2); norm.y(2) = 1/Kokkos::sqrt(2);
    tan2.z(2) = 1.0;

    transform_to_local_frame(a, norm, tan1, tan2);

    CHECK(Kokkos::abs(a.x(0) - 1.0) < 1e-14);
    CHECK(Kokkos::abs(a.y(0) - 1.0) < 1e-14);

    CHECK(Kokkos::abs(a.x(1) - 0.0) < 1e-14);
    CHECK(Kokkos::abs(a.y(1) - 1.0) < 1e-14);
}
