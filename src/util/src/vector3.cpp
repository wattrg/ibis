#include <cmath>
#include <cassert>
#include "vector3.h"

#include <doctest/doctest.h>
#include <Kokkos_MathematicalFunctions.hpp>

#define VEC3_TOL 1e-15

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
