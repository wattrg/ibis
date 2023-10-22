#ifndef VERTEX_H
#define VERTEX_H

#include "../../util/src/vector3.h"

template <typename T>
struct Vertex {
public:
    Vertex(Vector3<T> pos) : _pos(pos) {}
    Vector3<T> & pos() {return _pos;}

    bool operator == (const Vertex<T> &other) const {
        return _pos == other._pos;
    }

private:
    Vector3<T> _pos;
};

template <typename T>
struct Vertices {
public:
    Vertices () {}

    Vertices(int num_vertices);
    
    void set_vertex_position(int vertex_id, Vector3<T> &pos);

    KOKKOS_INLINE_FUNCTION
    Vector3s<T> &positions() {return _positions;}

    KOKKOS_INLINE_FUNCTION
    const Vector3s<T> &positions() const {return  _positions;}

    bool operator == (const Vertices &other) const {
        return _positions == other._positions;
    }
    
    KOKKOS_INLINE_FUNCTION
    int size() const {return _positions.size();}

private:
    Vector3s<T> _positions;
};

#endif
