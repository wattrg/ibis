#ifndef INTERFACE_H
#define INTERFACE_H

#include <unordered_map>
#include "../../util/src/id.h"
#include "../../util/src/field.h"
#include "../../util/src/vector3.h"
#include "grid_io.h"
#include "vertex.h"

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

    Interfaces(IdConstructor ids, std::vector<ElemType> shapes) 
        : m_vertex_ids(Id(ids)) 
    {
        shape_ = Field<ElemType>("Interface::shape", shapes.size());
        for (unsigned int i = 0; i < shapes.size(); i++) {
            shape_(i) = shapes[i];
        }

        norm_ = Vector3s<T>("Interface::norm", shapes.size());
        tan1_ = Vector3s<T>("Interface::tan1", shapes.size());
        tan2_ = Vector3s<T>("Interface::tan2", shapes.size());
        area_ = Field<T>("Interface::area", shapes.size());
    }

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
    Field<T> &area() const {return area_;}

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

    KOKKOS_FORCEINLINE_FUNCTION
    Vector3View<T> norm(const int i) const {
        return Vector3View(i, &norm_);
    } 

    KOKKOS_FORCEINLINE_FUNCTION
    Vector3View<T> &tan1(const int i) {
        return Vector3View(i, &tan1_);
    }

    KOKKOS_FORCEINLINE_FUNCTION
    Vector3View<T> &tan1(const int i) const {
        return Vector3View(i, &tan1_);
    }

    KOKKOS_FORCEINLINE_FUNCTION
    Vector3View<T> &tan2(const int i) {
        return Vector3View(i, &tan2_);
    }

    KOKKOS_FORCEINLINE_FUNCTION
    Vector3View<T> &tan2(const int i) const {
        return Vector3View(i, &tan2_);
    }

    KOKKOS_FORCEINLINE_FUNCTION
    int size() const {return m_vertex_ids.size();}

    void compute_orientations(Vertices<T> vertices) {
        // set the face tangents in parallel
        Kokkos::parallel_for("Interfaces::compute_orientations", norm_.size(), KOKKOS_LAMBDA (const int i){
            auto vertex_ids = m_vertex_ids[i];
            T x0 = vertices.position(vertex_ids(0), 0);
            T x1 = vertices.position(vertex_ids(1), 0);
            T y0 = vertices.position(vertex_ids(0), 1);
            T y1 = vertices.position(vertex_ids(1), 1);
            T z0 = vertices.position(vertex_ids(0), 2);
            T z1 = vertices.position(vertex_ids(1), 2);
            T ilength = 1./Kokkos::sqrt((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0) + (z1-z0));
            tan1_(i, 0) = ilength * (x1 - x0);
            tan1_(i, 1) = ilength * (y1 - y0);
            tan1_(i, 2) = ilength * (z1 - z0);

            switch (shape_(i)) {
                case ElemType::Line: {
                    auto vertex_ids = m_vertex_ids[i];
                    tan2_(i, 0) = 0.0;
                    tan2_(i, 1) = 0.0;
                    tan2_(i, 2) = 1.0;
                    break;
                }
                case ElemType::Tri: 
                    throw std::runtime_error("Not implemented"); 
                case ElemType::Quad:
                    throw std::runtime_error("Not implemented");
                default:
                    throw std::runtime_error("Invalid interface");
            }
        });

        // the face normal is the cross product of the tangents
        cross(tan1_, tan2_, norm_);
    }

    void compute_areas(Vertices<T> vertices) {
        Kokkos::parallel_for("Interfaces::compute_areas", area_.size(), KOKKOS_LAMBDA (const int i) {
            switch (shape_(i)) {
                case ElemType::Line: {
                    auto vertex_ids = m_vertex_ids[i];
                    T x1 = vertices.position(vertex_ids(0), 0);
                    T x2 = vertices.position(vertex_ids(1), 0);
                    T y1 = vertices.position(vertex_ids(0), 1);
                    T y2 = vertices.position(vertex_ids(1), 1);
                    area_(i) = Kokkos::sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1));
                    break;
                }
                case ElemType::Tri: {
                    auto vertex_ids = m_vertex_ids[i];
                    T x1 = vertices.position(vertex_ids(0), 0);
                    T x2 = vertices.position(vertex_ids(1), 0);
                    T x3 = vertices.position(vertex_ids(2), 0);
                    T y1 = vertices.position(vertex_ids(0), 1);
                    T y2 = vertices.position(vertex_ids(1), 1);
                    T y3 = vertices.position(vertex_ids(2), 1);
                    T area = 0.5*Kokkos::fabs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2));
                    area_(i) = area;
                    break;
                }
                case ElemType::Quad: {
                    auto vertex_ids = m_vertex_ids[i];
                    T x1 = vertices.position(vertex_ids(0), 0);
                    T x2 = vertices.position(vertex_ids(1), 0);
                    T x3 = vertices.position(vertex_ids(2), 0);
                    T x4 = vertices.position(vertex_ids(3), 0);
                    T y1 = vertices.position(vertex_ids(0), 1);
                    T y2 = vertices.position(vertex_ids(1), 1);
                    T y3 = vertices.position(vertex_ids(2), 1);
                    T y4 = vertices.position(vertex_ids(3), 1);
                    T area = x1*y2 + x2*y3 + x3*y4 + x4*y1 - x2*y1 - x3*y2 - x4*y3 - x1*y4;
                    area_(i) = 0.5 * Kokkos::fabs(area);
                    break;
                }
                case ElemType::Hex: {
                    throw std::runtime_error("Invalid interface"); 
                }
                case ElemType::Wedge: {
                    throw std::runtime_error("Invalid interface"); 
                }
                case ElemType::Pyramid: {
                    throw std::runtime_error("Invalid interface"); 
                }
            }
        });
    }

private:
    // the id's of the vertices forming each interface
    Id m_vertex_ids;

    // the cells to the left/right of the interface
    Field<int> left_cells_;
    Field<int> right_cells;

    // geometric data
    Field<T> area_;
    Field<ElemType> shape_;
    Vector3s<T> norm_;
    Vector3s<T> tan1_;
    Vector3s<T> tan2_;
    Vector3s<T> centre_;
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
