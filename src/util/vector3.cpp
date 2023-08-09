#include <cmath>
#include <cassert>
#include "vector3.h"

#include "../include/doctest/doctest/doctest.h"

using namespace Aeolus;

void Aeolus::dot(Vector3 &a, Vector3 &b, Field &result) {
    assert((a.extent(0) == b.extent(0)) && (b.extent(0) == result.extent(0)));
    
    Kokkos::parallel_for("vector3 dot product", a.extent(0), KOKKOS_LAMBDA(const int i){
        result(i) = a(i,0)*b(i,0) + a(i,1)*b(i,1) + a(i,2)*b(i,2);
    });
}

void Aeolus::add(Vector3 &a, Vector3 &b, Vector3 &result) {
    assert((a.extent(0) == b.extent(0)) && (b.extent(0) == result.extent(0)));

    Kokkos::parallel_for("Vector3 add", a.extent(0), KOKKOS_LAMBDA(const int i){
        result(i,0) = a(i,0) + b(i,0);
        result(i,1) = a(i,1) + b(i,1);
        result(i,2) = a(i,2) + b(i,2);
    });
}

void Aeolus::subtract(Vector3 &a, Vector3 &b, Vector3 &result) {
    assert((a.extent(0) == b.extent(0)) && (b.extent(0) == result.extent(0)));

    Kokkos::parallel_for("Vector3 subtract", a.extent(0), KOKKOS_LAMBDA(const int i) {
        result(i,0) = a(i,0) - b(i,0);
        result(i,1) = a(i,1) - b(i,1);
        result(i,2) = a(i,2) - b(i,2);
    });
}

void Aeolus::cross(Vector3 &a, Vector3 &b, Vector3 &result) {
    assert((a.extent(0) == b.extent(0)) && (b.extent(0) == result.extent(0)));

    Kokkos::parallel_for("Vector3 cross", a.extent(0), KOKKOS_LAMBDA(const int i) {
        result(i,0) = a(i,1)*b(i,2) - a(i,2)*b(i,1);
        result(i,1) = a(i,2)*b(i,0) - a(i,0)*b(i,2);
        result(i,2) = a(i,0)*b(i,1) - a(i,1)*b(i,0);
    });
}

void Aeolus::scale_in_place(Vector3 &a, double factor) {
    Kokkos::parallel_for("Vector3 scale", a.extent(0), KOKKOS_LAMBDA(const int i) {
        a(i, 0) *= factor;
        a(i, 1) *= factor;
        a(i, 2) *= factor;
    });
}

void Aeolus::length(Vector3 &a, Field &len) {
    assert(a.extent(0) == len.extent(0));

    Kokkos::parallel_for("Vector3 length", a.extent(0), KOKKOS_LAMBDA(const int i) {
        len(i) = sqrt(a(i,0)*a(i,0) + a(i,1)*a(i,1) + a(i,2)*a(i,2));
    });
}

void Aeolus::normalise(Vector3 &a) {
    Kokkos::parallel_for("Vector3 normalise", a.extent(0), KOKKOS_LAMBDA(const int i) {
        double length_inv = 1./sqrt(a(i,0)*a(i,0) + a(i,1)*a(i,1) + a(i,2)*a(i,2));
        a(i,0) *= length_inv;
        a(i,1) *= length_inv;
        a(i,2) *= length_inv;
    });
}

TEST_CASE("Vector Dot Product") {
    int n = 10;
    Aeolus::Vector3 a ("a", n);
    Aeolus::Vector3 b ("b", n);
    Aeolus::Field result ("result", n);
    Aeolus::Field expected ("expected", n);
    
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

    Aeolus::dot(a, b, result);

    // check the results are correct
    for (int i = 0; i < n; i++) {
        CHECK(fabs(expected(i) < result(i)) < 1e-8);
    }
}

TEST_CASE("Vector3 Add") {
    int n = 20;
    Aeolus::Vector3 a ("a", n);
    Aeolus::Vector3 b ("b", n);
    Aeolus::Vector3 result ("result", n);
    Aeolus::Vector3 expected ("expected", n);
    
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

    Aeolus::add(a, b, result);

    // check the results are correct
    for (int i = 0; i < n; i++) {
        CHECK(fabs(expected(i,0) - result(i,0)) < 1e-8);
        CHECK(fabs(expected(i,1) - result(i,1)) < 1e-8);
        CHECK(fabs(expected(i,2) - result(i,2)) < 1e-8);
    }
}

TEST_CASE("Vector3 subtract") {
    int n = 20;
    Aeolus::Vector3 a ("a", n);
    Aeolus::Vector3 b ("b", n);
    Aeolus::Vector3 result ("result", n);
    Aeolus::Vector3 expected ("expected", n);
    
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

    Aeolus::subtract(a, b, result);

    // check the results are correct
    for (int i = 0; i < n; i++) {
        CHECK(fabs(expected(i,0) - result(i,0)) < 1e-8);
        CHECK(fabs(expected(i,1) - result(i,1)) < 1e-8);
        CHECK(fabs(expected(i,2) - result(i,2)) < 1e-8);
    }
}

TEST_CASE("Vector3 cross") {
    int n = 20;
    Aeolus::Vector3 a ("a", n);
    Aeolus::Vector3 b ("b", n);
    Aeolus::Vector3 result ("result", n);
    Aeolus::Vector3 expected ("expected", n);
    
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

    Aeolus::cross(a, b, result);

    // check the results are correct
    for (int i = 0; i < n; i++) {
        CHECK(fabs(expected(i,0) - result(i,0)) < 1e-8);
        CHECK(fabs(expected(i,1) - result(i,1)) < 1e-8);
        CHECK(fabs(expected(i,2) - result(i,2)) < 1e-8);
    }
}

TEST_CASE("Vector3 scale_in_place") {
    int n = 20;
    Aeolus::Vector3 a ("a", n);
    double factor = 2.0;

    for (int i = 0; i < n; i++){
        a(i, 0) = 1.0 * i;
        a(i, 1) = 2.0 * i;
        a(i, 2) = 3.0 * i;
    }

    Aeolus::scale_in_place(a, factor);

    for (int i = 0; i < n; i++){
        CHECK(fabs(a(i, 0) - 2.0 * i) < 1e-8);
        CHECK(fabs(a(i, 1) - 4.0 * i) < 1e-8);
        CHECK(fabs(a(i, 2) - 6.0 * i) < 1e-8);
    }
}

TEST_CASE("Vector3 length") {
    int n = 20;
    Aeolus::Vector3 a ("a", n);
    Aeolus::Field len ("length", n);
    Aeolus::Field expected ("expected", n);

    for (int i = 0; i < n; i++){
        a(i, 0) = 1.0 * i;
        a(i, 1) = 2.0 * i;
        a(i, 2) = 3.0 * i;

        expected(i) = sqrt(a(i,0)*a(i,0) + a(i,1)*a(i,1) + a(i,2)*a(i,2));
    }

    Aeolus::length(a, len);

    for (int i = 0; i < n; i++){
        CHECK(fabs(len(i) - expected(i)) < 1e-8);
    }
}

TEST_CASE("Vector3 normalise") {
    int n = 20;
    Aeolus::Vector3 a ("a", n);

    for (int i = 0; i < n; i++){
        a(i, 0) = 1.0 * i + 1.0;
        a(i, 1) = 2.0 * i;
        a(i, 2) = 3.0 * i;
    }

    Aeolus::normalise(a);

    for (int i = 0; i < n; i++){
        double length_inv = sqrt((i+1.0)*(i+1.0) + 4.0*i*i + 9.0*i*i);
        CHECK(fabs(a(i,0) - 1.0 * (i+1.0) / length_inv) < 1e-8);
        CHECK(fabs(a(i,1) - 2.0 * i / length_inv) < 1e-8);
        CHECK(fabs(a(i,2) - 3.0 * i / length_inv) < 1e-8);
    }
}
