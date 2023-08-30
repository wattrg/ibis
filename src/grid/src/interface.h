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

    inline auto vertex_ids() {return _interfaces->vertex_ids()[_id];}
    inline T& area() const {return _interfaces->area(_id);}

private:
    Interfaces<T> * _interfaces;
    int _id;
};

template <typename T>
struct Interfaces {
public:
    Interfaces () {}

    Interfaces(IdConstructor ids, std::vector<ElemType> shapes) 
        : _vertex_ids(Id(ids)) 
    {
        _shape = Field<ElemType>("Interface::shape", shapes.size());
        for (unsigned int i = 0; i < shapes.size(); i++) {
            _shape(i) = shapes[i];
        }

        _norm = Vector3s<T>("Interface::norm", shapes.size());
        _tan1 = Vector3s<T>("Interface::tan1", shapes.size());
        _tan2 = Vector3s<T>("Interface::tan2", shapes.size());
        _area = Field<T>("Interface::area", shapes.size());
    }

    bool operator == (const Interfaces &other) const {
        return _vertex_ids == other._vertex_ids;
    }

    inline InterfaceView<T> operator[] (const int i) {
        assert(i < size());
        return InterfaceView<T>(this, i);
    }

    inline Id &vertex_ids() {return _vertex_ids;}

    inline Field<T> &area() const {return _area;}
    inline T& area(int i) const {return _area(i);}
    inline T& area(int i) {return _area(i);}

    inline Vector3s<T> norm() {return _norm;}
    inline Vector3s<T> tan1() {return _tan1;}
    inline Vector3s<T> tan2() {return _tan2;}

    inline Vector3View<T> norm(const int i) {
        return Vector3View(i, &_norm);
    } 
    inline Vector3View<T> norm(const int i) const {
        return Vector3View(i, &_norm);
    } 

    inline Vector3View<T> &tan1(const int i) {
        return Vector3View(i, &_tan1);
    }
    inline Vector3View<T> &tan1(const int i) const {
        return Vector3View(i, &_tan1);
    }

    inline Vector3View<T> &tan2(const int i) {
        return Vector3View(i, &_tan2);
    }
    inline Vector3View<T> &tan2(const int i) const {
        return Vector3View(i, &_tan2);
    }

    inline int size() const {return _vertex_ids.size();}

    void compute_orientations(Vertices<T> vertices) {
        // set the face tangents in parallel
        Kokkos::parallel_for("Interfaces::compute_orientations", _norm.size(), KOKKOS_LAMBDA (const int i){
            auto vertex_ids = _vertex_ids[i];
            T x0 = vertices.position(vertex_ids(0), 0);
            T x1 = vertices.position(vertex_ids(1), 0);
            T y0 = vertices.position(vertex_ids(0), 1);
            T y1 = vertices.position(vertex_ids(1), 1);
            T z0 = vertices.position(vertex_ids(0), 2);
            T z1 = vertices.position(vertex_ids(1), 2);
            T ilength = 1./Kokkos::sqrt((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0) + (z1-z0));
            _tan1(i, 0) = ilength * (x1 - x0);
            _tan1(i, 1) = ilength * (y1 - y0);
            _tan1(i, 2) = ilength * (z1 - z0);

            switch (_shape(i)) {
                case ElemType::Line: {
                    auto vertex_ids = _vertex_ids[i];
                    _tan2(i, 0) = 0.0;
                    _tan2(i, 1) = 0.0;
                    _tan2(i, 2) = 1.0;
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
        cross(_tan1, _tan2, _norm);
    }

    void compute_areas(Vertices<T> vertices) {
        Kokkos::parallel_for("Interfaces::compute_areas", _area.size(), KOKKOS_LAMBDA (const int i) {
            switch (_shape(i)) {
                case ElemType::Line: {
                    auto vertex_ids = _vertex_ids[i];
                    T x1 = vertices.position(vertex_ids(0), 0);
                    T x2 = vertices.position(vertex_ids(1), 0);
                    T y1 = vertices.position(vertex_ids(0), 1);
                    T y2 = vertices.position(vertex_ids(1), 1);
                    _area(i) = Kokkos::sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1));
                    break;
                }
                case ElemType::Tri: {
                    auto vertex_ids = _vertex_ids[i];
                    T x1 = vertices.position(vertex_ids(0), 0);
                    T x2 = vertices.position(vertex_ids(1), 0);
                    T x3 = vertices.position(vertex_ids(2), 0);
                    T y1 = vertices.position(vertex_ids(0), 1);
                    T y2 = vertices.position(vertex_ids(1), 1);
                    T y3 = vertices.position(vertex_ids(2), 1);
                    T area = 0.5*Kokkos::fabs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2));
                    _area(i) = area;
                    break;
                }
                case ElemType::Quad: {
                    auto vertex_ids = _vertex_ids[i];
                    T x1 = vertices.position(vertex_ids(0), 0);
                    T x2 = vertices.position(vertex_ids(1), 0);
                    T x3 = vertices.position(vertex_ids(2), 0);
                    T x4 = vertices.position(vertex_ids(3), 0);
                    T y1 = vertices.position(vertex_ids(0), 1);
                    T y2 = vertices.position(vertex_ids(1), 1);
                    T y3 = vertices.position(vertex_ids(2), 1);
                    T y4 = vertices.position(vertex_ids(3), 1);
                    T area = x1*y2 + x2*y3 + x3*y4 + x4*y1 - x2*y1 - x3*y2 - x4*y3 - x1*y4;
                    _area(i) = 0.5 * Kokkos::fabs(area);
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
    Id _vertex_ids;

    // geometric data
    Field<T> _area;
    Field<ElemType> _shape;
    Vector3s<T> _norm;
    Vector3s<T> _tan1;
    Vector3s<T> _tan2;
    Vector3s<T> _centre;
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
    std::unordered_map<std::string, int> _hash_map;

    std::string _hash(std::vector<int> vertex_ids);
    bool _contains(std::string hash);
};

#endif
