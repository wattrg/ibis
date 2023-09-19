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

    Vertices(int num_vertices) {
        _positions = Vector3s<T>("vertices", num_vertices);
    }
    
    // Vertices(std::vector<Vertex<T>> vertices);

    void set_vertex_position(int vertex_id, Vector3<T> &pos){
        _positions(vertex_id, 0) = pos.x;
        _positions(vertex_id, 1) = pos.y;
        _positions(vertex_id, 2) = pos.z;
    }

    Vector3s<T> &positions() {return _positions;}
    const Vector3s<T> &positions() const {return  _positions;}

    T& position(int i, int coordinate) {return _positions(i, coordinate);}
    T& position(int i, int coordinate) const {return _positions(i, coordinate);}

    Vector3View<T> position(int i) {return Vector3View<T>(i, &_positions);}
    Vector3View<T> position(int i) const {return Vector3View<T>(i, &_positions);}

    bool operator == (const Vertices &other) const {
        return _positions == other._positions;
    }

private:
    Vector3s<T> _positions;
};

#endif
