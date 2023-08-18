#ifndef CELL_H
#define CELL_H

#include "../../util/src/id.h"
#include "../../util/src/field.h"
#include "grid_io.h"

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
        _shape = Field<ElemType>("shape", shapes.size());
        for (unsigned int i = 0; i < shapes.size(); i++) {
            _shape(i) = shapes[i]; 
        }
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

    void compute_volumes();

private:
    Id _interface_ids;
    Id _vertex_ids;
    Field<ElemType> _shape;
    Field<T> _volume;
    Field<int> _outsign;
};

#endif
