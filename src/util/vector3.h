#ifndef VECTOR3_H
#define VECTOR3_H

#include "field.h"

struct Vector3 {
public:
    Vector3(int n);
    ~Vector3();
   
    inline int length() {return _length;}
    inline double* x() {return _x;}
    inline double* y() {return _y;}
    inline double* z() {return _z;}

    void dot(Vector3 &other, Field &result);
    
private:
    int _length;
    double *_x;
    double *_y;
    double *_z;
};

#endif
