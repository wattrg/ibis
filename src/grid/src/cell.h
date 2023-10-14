#ifndef CELL_H
#define CELL_H

#include <Kokkos_Core.hpp>
#include "Kokkos_Macros.hpp"
#include "Kokkos_MathematicalFunctions.hpp"
#include "../../util/src/id.h"
#include "../../util/src/field.h"
#include "grid_io.h"
#include "vertex.h"

template <typename T> struct Cells;

template <typename T>
struct CellView {
public:
    CellView(Cells<T> *cells_, int id_) 
        : cells_(cells_), id_(id_) {}

    KOKKOS_FORCEINLINE_FUNCTION
    auto vertex_ids() {return cells_->vertex_ids()[id_];}

private:
    Cells<T> *cells_;
    int id_;
};

template <typename T>
struct CellFaces {
public:
    CellFaces() {}

    CellFaces(const Id& interface_ids);

    bool operator == (const CellFaces& other) const;

    KOKKOS_INLINE_FUNCTION
    auto face_ids(const int i) const {
        int first = offsets_(i);
        int last = offsets_(i+1);
        return Kokkos::subview(face_ids_, std::make_pair(first, last));
    }

    KOKKOS_INLINE_FUNCTION
    auto outsigns(const int i) const {
        int first = offsets_(i);
        int last = offsets_(i+1);
        return Kokkos::subview(outsigns_, std::make_pair(first, last));
    }

    void set_outsign(const int i_cell, const int i_face, const int sign) {
        outsigns(i_cell)(i_face) = sign;
    }

private:
    Kokkos::View<int*> offsets_;
    Kokkos::View<int*> face_ids_;
    Kokkos::View<int*> outsigns_;
};

template <typename T>
struct Cells {
public:
    Cells () {}

    Cells(Id vertices, Id interfaces, std::vector<ElemType> shapes);

    bool operator == (const Cells &other) const {
        return (faces_ == other.faces_) &&
               (vertex_ids_ == other.vertex_ids_);
    }

    KOKKOS_FORCEINLINE_FUNCTION
    CellView<T> operator[] (const int i) {
        assert(i < size());
        return CellView<T>(this, i);
    }

    KOKKOS_FORCEINLINE_FUNCTION
    const Id &vertex_ids() const {return vertex_ids_;}

    // KOKKOS_FORCEINLINE_FUNCTION
    // const Id &interface_ids() const {return interface_ids_;}
    
    KOKKOS_FORCEINLINE_FUNCTION
    int size() const {return num_cells_;}

    KOKKOS_FORCEINLINE_FUNCTION
    const T& volume(const int i) const {return volume_(i);}

    const Vector3s<T>& centroids() const {return centroid_;}

    void compute_centroids(const Vertices<T>& vertices);

    void compute_volumes(const Vertices<T>& vertices);

    CellFaces<T> faces() const {return faces_;}

    const Field<ElemType>& shapes() const {return shape_;}

private:
    CellFaces<T> faces_;
    // Id interface_ids_;
    Id vertex_ids_;
    Field<ElemType> shape_;
    Field<T> volume_;
    // Id outsign_;
    Vector3s<T> centroid_;

    int num_cells_;
};

#endif
