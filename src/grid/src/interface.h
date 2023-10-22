#ifndef INTERFACE_H
#define INTERFACE_H

#include <unordered_map>
#include "../../util/src/id.h"
#include "../../util/src/field.h"
#include "../../util/src/vector3.h"
#include "Kokkos_Macros.hpp"
#include "grid_io.h"
#include "vertex.h"
#include "cell.h"

template <typename T> struct Interfaces;

// template <typename T>
// struct InterfaceView {
// public:
//     InterfaceView(Interfaces<T> *interfaces, int id) 
//         : _interfaces(interfaces), _id(id) {}
//
//     KOKKOS_FORCEINLINE_FUNCTION 
//     auto vertex_ids() {return _interfaces->vertex_ids()[_id];}
//
//     KOKKOS_FORCEINLINE_FUNCTION 
//     T& area() const {return _interfaces->area(_id);}
//
// private:
//     Interfaces<T> * _interfaces;
//     int _id;
// };

template <typename T>
struct Interfaces {
public:
    Interfaces () {}

    Interfaces(IdConstructor ids, std::vector<ElemType> shapes);

    bool operator == (const Interfaces &other) const {
        return vertex_ids_ == other.vertex_ids_;
    }

    // KOKKOS_INLINE_FUNCTION 
    // InterfaceView<T> operator[] (const int i) {
    //     assert(i < size());
    //     return InterfaceView<T>(this, i);
    // }

    KOKKOS_INLINE_FUNCTION 
    Id<> &vertex_ids() {return vertex_ids_;}

    KOKKOS_INLINE_FUNCTION
    const Field<T> &area() const {return area_;}

    KOKKOS_INLINE_FUNCTION
    T& area(int i) const {return area_(i);}

    KOKKOS_INLINE_FUNCTION
    T& area(int i) {return area_(i);}

    KOKKOS_INLINE_FUNCTION
    Vector3s<T>& norm() {return norm_;}

    KOKKOS_INLINE_FUNCTION
    const Vector3s<T>& norm() const {return norm_;}

    KOKKOS_INLINE_FUNCTION
    Vector3s<T>& tan1() {return tan1_;}

    KOKKOS_INLINE_FUNCTION
    const Vector3s<T>& tan1() const {return tan1_;}

    KOKKOS_INLINE_FUNCTION
    Vector3s<T>& tan2() {return tan2_;}

    KOKKOS_INLINE_FUNCTION
    const Vector3s<T>& tan2() const {return tan2_;}

    KOKKOS_INLINE_FUNCTION
    Vector3s<T>& centre() {return centre_;}

    KOKKOS_INLINE_FUNCTION
    const Vector3s<T>& centre() const {return centre_;}

    KOKKOS_INLINE_FUNCTION
    void attach_cell_left(const int cell_id,  const int face_id) const {
        left_cells_(face_id) = cell_id;
    }

    KOKKOS_INLINE_FUNCTION
    void attach_cell_right(const int cell_id, const int face_id) const {
        right_cells_(face_id) = cell_id;
    }

    // KOKKOS_FORCEINLINE_FUNCTION
    // Vector3View<T> norm(const int i) {
    //     return Vector3View(i, &norm_);
    // } 

    KOKKOS_INLINE_FUNCTION
    int left_cell(const int face_id) const {return left_cells_(face_id);}

    KOKKOS_INLINE_FUNCTION
    int right_cell(const int face_id) const {return right_cells_(face_id);}


    // KOKKOS_FORCEINLINE_FUNCTION
    // Vector3View<T> norm(const int i) const {
    //     return Vector3View(i, &norm_);
    // } 

    // KOKKOS_FORCEINLINE_FUNCTION
    // Vector3View<T> &tan1(const int i) {
    //     return Vector3View(i, &tan1_);
    // }

    // KOKKOS_FORCEINLINE_FUNCTION
    // Vector3View<T> &tan1(const int i) const {
    //     return Vector3View(i, &tan1_);
    // }

    // KOKKOS_FORCEINLINE_FUNCTION
    // Vector3View<T> &tan2(const int i) {
    //     return Vector3View(i, &tan2_);
    // }

    // KOKKOS_FORCEINLINE_FUNCTION
    // Vector3View<T> &tan2(const int i) const {
    //     return Vector3View(i, &tan2_);
    // }

    KOKKOS_INLINE_FUNCTION
    int size() const {return vertex_ids_.size();}

    void compute_orientations(Vertices<T> vertices);

    void compute_areas(Vertices<T> vertices);

    void compute_centres(Vertices<T> vertices);


private:
    int size_;
    // the id's of the vertices forming each interface
    Id<> vertex_ids_;

    // the cells to the left/right of the interface
    Field<int> left_cells_;
    Field<int> right_cells_;

    // geometric data
    Field<T> area_;
    Field<ElemType> shape_;
    Vector3s<T> norm_;
    Vector3s<T> tan1_;
    Vector3s<T> tan2_;
    Vector3s<T> centre_;

    // boundary informaton
    Field<bool> on_boundary_;
};


// Efficient look-up of interface ID 
// from the index of the vertices
// forming the interface
struct InterfaceLookup {
public:
    InterfaceLookup();

    int insert(std::vector<int> vertex_ids);
    bool contains(std::vector<int> vertex_ids);
    int id(std::vector<int> vertex_ids); 

private:
    std::unordered_map<std::string, int> hash_map_;

    std::string hash_vertex_ids(std::vector<int> vertex_ids);
    bool contains_hash(std::string hash);
};

#endif
