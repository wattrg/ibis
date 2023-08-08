#include "field.h"

Field::Field(int n) : _length(n) {
    _data = new double[n];
}

Field::~Field() {
    delete _data;
}
