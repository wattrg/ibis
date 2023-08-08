#include <cassert>
#include "vector3.h"

#include "../include/doctest/doctest/doctest.h"

Vector3::Vector3(int n) : _length(n) {
    _x = new double[n];
    _y = new double[n];
    _z = new double[n];
}

Vector3::~Vector3() {
    delete _x;
    delete _y;
    delete _z;
}

void Vector3::dot(Vector3 &other, Field &result) {
    assert(this->_length == other._length);
    assert(this->_length == result.length());

    for (int i=0; i < this->_length; i++){
        result[i] = _x[i]*other._x[i] + _y[i]*other._y[i] + _z[i]*other._z[i];
    }
}

TEST_CASE("test test case") {
    CHECK(1 == 1);
}
