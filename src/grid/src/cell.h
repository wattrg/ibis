#ifndef CELL_H
#define CELL_H

#include <Kokkos_Core.hpp>
#include "Kokkos_MathematicalFunctions.hpp"
#include "../../util/src/id.h"
#include "../../util/src/field.h"
#include "grid_io.h"
#include "vertex.h"

template <typename T> struct Cells;

template <typename T>
struct CellView {
public:
    CellView(Cells<T> *cells, int id) 
        : _cells(cells), _id(id) {}

    inline auto vertex_ids() {return _cells->vertex_ids()[_id];}

private:
    Cells<T> *_cells;
    int _id;
};

template <typename T>
struct Cells {
public:
    Cells () {}

    Cells(Id vertices, Id interfaces, std::vector<ElemType> shapes) 
        : _interface_ids(interfaces), _vertex_ids(vertices) 
    {
        _shape = Field<ElemType>("Cell::shape", shapes.size());
        for (unsigned int i = 0; i < shapes.size(); i++) {
            _shape(i) = shapes[i]; 
        }

        _volume = Field<T>("Cell::Volume", shapes.size());
    }

    bool operator == (const Cells &other) const {
        return (_interface_ids == other._interface_ids) &&
               (_vertex_ids == other._vertex_ids);
    }

    inline CellView<T> operator[] (const int i) {
        assert(i < size());
        return CellView<T>(this, i);
    }

    inline Id &vertex_ids() {return _vertex_ids;}
    inline Id &interface_ids() {return _interface_ids;}

    inline int size() const {return _interface_ids.size();}

    T & volume(const int i) const {return _volume(i);}

    void compute_volumes(Vertices<T> vertices) {
        // TODO: It would be nicer to move each case in the switch 
        // to a function sitting somewhere else to keep the amount
        // of code in this method down, and avoid duplication with
        // computing the area of interfaces. However, this won't
        // be trivial for the GPU.
        Kokkos::parallel_for("Cells::compute_volume", _volume.size(), KOKKOS_LAMBDA(const int i) {
            switch (_shape(i)) {
                case ElemType::Line:
                    throw std::runtime_error("Invalid cell shape");
                    break;
                case ElemType::Tri: {
                    auto vertex_ids = _vertex_ids[i];
                    T x1 = vertices.position(vertex_ids(0), 0);
                    T x2 = vertices.position(vertex_ids(1), 0);
                    T x3 = vertices.position(vertex_ids(2), 0);
                    T y1 = vertices.position(vertex_ids(0), 1);
                    T y2 = vertices.position(vertex_ids(1), 1);
                    T y3 = vertices.position(vertex_ids(2), 1);
                    T area = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2);
                    _volume(i) = 0.5 * Kokkos::fabs(area);
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
                    T area = x1*y2 + x2*y3 + x3*y4 + x4*y1 - 
                                 x2*y1 - x3*y2 - x4*y3 - x1*y4;
                    _volume(i) = 0.5 * Kokkos::fabs(area);
                    break;
                }
                case ElemType::Hex:
                    throw std::runtime_error("Not implemented");
                    break;
                case ElemType::Wedge:
                    throw std::runtime_error("Not implemented");
                    break;
                case ElemType::Pyramid:
                    throw std::runtime_error("Not implemented");
                    break;
            }
        }); 
    }

private:
    Id _interface_ids;
    Id _vertex_ids;
    Field<ElemType> _shape;
    Field<T> _volume;
    Field<int> _outsign;
};

#endif
