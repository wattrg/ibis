#ifndef VECTOR3_H
#define VECTOR3_H

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include "Kokkos_Macros.hpp"
#include "field.h"

// A single vector with 3 components
template <typename T>
struct Vector3 {
    Vector3(){}

    Vector3(T x, T y, T z) : x(x), y(y), z(z) {}
    T x, y, z;

    bool operator == (const Vector3 &other) const {
        return (std::fabs(x - other.x) < 1e-14) &&
               (std::fabs(y - other.y) < 1e-14) &&
               (std::fabs(z - other.z) < 1e-14);
    }
};

template <typename T>
struct Vector3s;

// template <typename T>
// struct Vector3View {
// public:
//     Vector3View(int index, Vector3s<T> * vectors) : _index(index), _vectors(vectors) {}
//     
//     inline T & x() const {return _vectors(_index, 0);}
//     inline T & x() {return (*_vectors)(_index, 0);}
// 
//     inline T & y() const {return _vectors(_index, 1);}
//     inline T & y() {return (*_vectors)(_index, 1);}
// 
//     inline T & z() const {return _vectors(_index, 2);}
//     inline T & z() {return (*_vectors)(_index, 2);}
// 
// private:
//     int _index;
//     Vector3s<T> *_vectors;
// };

template <typename T>
struct Vector3s {
public:
    Vector3s() {}

    // ~Vector3s(){}

    Vector3s(std::string description, int n) {
        view_ = Kokkos::View<T*[3]>(description, n);
    }

    Vector3s(int n) {
        view_ = Kokkos::View<T*[3]>("Vector3s", n);
    }

    KOKKOS_FORCEINLINE_FUNCTION
    T& operator() (const int i, const int j) {
        return view_(i, j); 
    }

    KOKKOS_FORCEINLINE_FUNCTION 
    T& operator() (const int i, const int j) const {
        return view_(i, j);
    }

    // KOKKOS_FORCEINLINE_FUNCTION
    // Vector3View<T> operator[] (const int i) {
    //     return Vector3View<T> (i, this);
    // }
    //
    // KOKKOS_FORCEINLINE_FUNCTION
    // Vector3View<T> operator[] (const int i) const {
    //     return Vector3View<T> (i, this);
    // }

    KOKKOS_FORCEINLINE_FUNCTION
    T& x(const int i) {return view_(i, 0);}

    KOKKOS_FORCEINLINE_FUNCTION
    T& x(const int i) const {return view_(i, 0);}

    KOKKOS_FORCEINLINE_FUNCTION
    T& y(const int i) {return view_(i, 1);}

    KOKKOS_FORCEINLINE_FUNCTION
    T& y(const int i) const {return view_(i, 1);}

    KOKKOS_FORCEINLINE_FUNCTION
    T& z(const int i) {return view_(i, 2);}

    KOKKOS_FORCEINLINE_FUNCTION
    T& z(const int i) const {return view_(i, 2);}

    KOKKOS_FORCEINLINE_FUNCTION
    void copy_vector(const Vector3<T>& vector, const int i) {
        x(i) = vector.x;
        y(i) = vector.y;
        z(i) = vector.z;
    }

    KOKKOS_FORCEINLINE_FUNCTION 
    int size() const {return view_.extent(0);}

    KOKKOS_FORCEINLINE_FUNCTION
    bool operator == (const Vector3s &other) const {
        if (this->size() != other.size()) {
            return false;
        }

        for (int i = 0; i < this->size(); i++) {
            if (fabs(view_(i, 0) - other.view_(i, 0)) > 1e-14)  return false;
            if (fabs(view_(i, 1) - other.view_(i, 1)) > 1e-14)  return false;
            if (fabs(view_(i, 2) - other.view_(i, 2)) > 1e-14)  return false;
        }
        return true;
    }


private:
    Kokkos::View<T*[3]> view_;
};

template <typename T>
KOKKOS_INLINE_FUNCTION
T dot(const Vector3s<T>& a, const Vector3s<T>& b, const int i) {
    return a.x(i)*b.x(i) + a.y(i)*b.y(i) + a.z(i)*b.z(i);
}

template <typename T>
KOKKOS_INLINE_FUNCTION
void cross(const Vector3s<T>& a, const Vector3s<T>& b, const Vector3s<T>& c, const int i){
    c.x(i) = a.y(i)*b.z(i) - a.z(i)*b.y(i);
    c.y(i) = a.z(i)*b.x(i) - a.x(i)*b.z(i);
    c.z(i) = a.x(i)*b.y(i) - a.y(i)*b.x(i);
}

template <typename T>
void dot(const Vector3s<T> &a, const Vector3s<T> &b, Field<T> &result);


template <typename T>
void add(const Vector3s<T> &a, const Vector3s<T> &b, Vector3s<T> &result);


template <typename T>
void subtract(const Vector3s<T> &a, const Vector3s<T> &b, Vector3s<T> &result);

template <typename T>
void cross(const Vector3s<T> &a, const Vector3s<T> &b, Vector3s<T> &result);

template <typename T>
void scale_in_place(Vector3s<T> &a, T factor);

template <typename T>
void length(const Vector3s<T> &a, Field<T> &len);

template <typename T>
void normalise(Vector3s<T> &a);

template <typename T>
void transform_to_local_frame(Vector3s<T>& a, const Vector3s<T>& norm, 
                              const Vector3s<T>& tan1, const Vector3s<T> tan2);

template <typename T>
void transform_to_global_frame(Vector3s<T>& a, const Vector3s<T>& norm, 
                               const Vector3s<T>& tan1, const Vector3s<T>& tan2);



#endif
