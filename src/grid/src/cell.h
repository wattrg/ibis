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
struct Cells {
public:
    Cells () {}

    Cells(Id vertices, Id interfaces, std::vector<ElemType> shapes);
    //     : interface_ids_(interfaces), vertex_ids_(vertices) 
    // {
    //     shape_ = Field<ElemType>("Cell::shape", shapes.size());
    //     for (unsigned int i = 0; i < shapes.size(); i++) {
    //         shape_(i) = shapes[i]; 
    //     }
    //
    //     volume_ = Field<T>("Cell::Volume", shapes.size());
    //     centroid_ = Vector3s<T>("Cell::centroids", shapes.size());
    // }

    bool operator == (const Cells &other) const {
        return (interface_ids_ == other.interface_ids_) &&
               (vertex_ids_ == other.vertex_ids_);
    }

    KOKKOS_FORCEINLINE_FUNCTION
    CellView<T> operator[] (const int i) {
        assert(i < size());
        return CellView<T>(this, i);
    }

    KOKKOS_FORCEINLINE_FUNCTION
    const Id &vertex_ids() const {return vertex_ids_;}

    KOKKOS_FORCEINLINE_FUNCTION
    const Id &interface_ids() const {return interface_ids_;}

    KOKKOS_FORCEINLINE_FUNCTION
    int size() const {return interface_ids_.size();}

    KOKKOS_FORCEINLINE_FUNCTION
    const T& volume(const int i) const {return volume_(i);}

    const Vector3s<T>& centroids() const {return centroid_;}

    void compute_centroids(const Vertices<T>& vertices);

    void compute_volumes(const Vertices<T>& vertices);

private:
    Id interface_ids_;
    Id vertex_ids_;
    Field<ElemType> shape_;
    Field<T> volume_;
    Field<int> outsign_;
    Vector3s<T> centroid_;
};

#endif
