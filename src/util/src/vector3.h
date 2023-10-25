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

template <typename T, 
          class Layout=Kokkos::DefaultExecutionSpace::array_layout,
          class Space=Kokkos::DefaultExecutionSpace::memory_space>
struct Vector3s {
public:
    using view_type = Kokkos::View<T*[3], Layout, Space>;
    using array_layout = typename view_type::array_layout;
    using memory_space = typename view_type::memory_space;
    using mirror_view_type = typename view_type::host_mirror_type;
    using mirror_layout = typename mirror_view_type::array_layout;
    using mirror_space = typename mirror_view_type::memory_space;
    using mirror_type = Vector3s<T, mirror_layout, mirror_space>;

public:
    Vector3s() {}

    // ~Vector3s(){}

    Vector3s(std::string description, int n) {
        view_ = view_type(description, n);
    }

    Vector3s(int n) {
        view_ = view_type("Vector3s", n);
    }

    KOKKOS_INLINE_FUNCTION
    T& operator() (const int i, const int j) {
        return view_(i, j); 
    }

    KOKKOS_INLINE_FUNCTION 
    T& operator() (const int i, const int j) const {
        return view_(i, j);
    }

    KOKKOS_INLINE_FUNCTION
    T& x(const int i) {return view_(i, 0);}

    KOKKOS_INLINE_FUNCTION
    T& x(const int i) const {return view_(i, 0);}

    KOKKOS_INLINE_FUNCTION
    T& y(const int i) {return view_(i, 1);}

    KOKKOS_INLINE_FUNCTION
    T& y(const int i) const {return view_(i, 1);}

    KOKKOS_INLINE_FUNCTION
    T& z(const int i) {return view_(i, 2);}

    KOKKOS_INLINE_FUNCTION
    T& z(const int i) const {return view_(i, 2);}

    KOKKOS_INLINE_FUNCTION
    void copy_vector(const Vector3<T>& vector, const int i) {
        x(i) = vector.x;
        y(i) = vector.y;
        z(i) = vector.z;
    }

    KOKKOS_INLINE_FUNCTION 
    int size() const {return view_.extent(0);}

    KOKKOS_INLINE_FUNCTION
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

    mirror_type host_mirror() {
        return mirror_type(view_.extent(0));
    }

    template <class OtherSpace>
    void deep_copy(const Vector3s<T, Layout, OtherSpace>& other) {
        Kokkos::deep_copy(view_, other.view_);
    }

public:
    view_type view_;
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
