#ifndef INTERFACE_H
#define INTERFACE_H

#include <unordered_map>
#include "../../util/src/id.h"
#include "../../util/src/field.h"
#include "../../util/src/vector3.h"
#include "grid_io.h"
#include "vertex.h"
#include "cell.h"

template <typename T> struct Interfaces;

template <typename T>
struct InterfaceView {
public:
    InterfaceView(Interfaces<T> *interfaces, int id) 
        : _interfaces(interfaces), _id(id) {}

    KOKKOS_FORCEINLINE_FUNCTION auto vertex_ids() {return _interfaces->vertex_ids()[_id];}
    KOKKOS_FORCEINLINE_FUNCTION T& area() const {return _interfaces->area(_id);}

private:
    Interfaces<T> * _interfaces;
    int _id;
};

template <typename T>
struct Interfaces {
public:
    Interfaces () {}

    Interfaces(IdConstructor ids, std::vector<ElemType> shapes);

    bool operator == (const Interfaces &other) const {
        return m_vertex_ids == other.m_vertex_ids;
    }

    KOKKOS_FORCEINLINE_FUNCTION InterfaceView<T> operator[] (const int i) {
        assert(i < size());
        return InterfaceView<T>(this, i);
    }

    KOKKOS_FORCEINLINE_FUNCTION 
    Id &vertex_ids() {return m_vertex_ids;}

    KOKKOS_FORCEINLINE_FUNCTION
    const Field<T> &area() const {return area_;}

    KOKKOS_FORCEINLINE_FUNCTION
    T& area(int i) const {return area_(i);}

    KOKKOS_FORCEINLINE_FUNCTION
    T& area(int i) {return area_(i);}

    KOKKOS_FORCEINLINE_FUNCTION
    Vector3s<T> norm() {return norm_;}

    KOKKOS_FORCEINLINE_FUNCTION
    Vector3s<T> norm() const {return norm_;}

    KOKKOS_FORCEINLINE_FUNCTION
    Vector3s<T> tan1() {return tan1_;}

    KOKKOS_FORCEINLINE_FUNCTION
    Vector3s<T> tan2() {return tan2_;}

    KOKKOS_FORCEINLINE_FUNCTION
    Vector3View<T> norm(const int i) {
        return Vector3View(i, &norm_);
    } 

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

    KOKKOS_FORCEINLINE_FUNCTION
    int size() const {return m_vertex_ids.size();}

    void compute_orientations(Vertices<T> vertices);

    void compute_areas(Vertices<T> vertices);

    void mark_on_boundary(const int i);

    // void compute_connectivity(const Vertices<T>& vertices, const Cells<T>& cells){
    //     (void) vertices;
    //     Id& face_ids = cells.interface_ids();
    //     for (int cell_i = 0; cell_i < cells.size(); cell_i++){
    //         auto cell_face_ids = face_ids[cell_i];
    //         for (int face_idx = 0; face_idx < cell_face_ids.size(); face_idx++){
    //             int face_id = cell_face_ids(face_idx);
    //             
    //             if (left_cells_(face_id) == -1 || right_cells_(face_id) == -1) {
    //                 // 
    //             } 
    //         }
    //     }    
    // }

private:
    int size_;
    // the id's of the vertices forming each interface
    Id m_vertex_ids;

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
