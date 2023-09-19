#include "vertex.h"

template <typename T>
Vertices<T>::Vertices(int num_vertices) {
    _positions = Vector3s<T>("Vertices", num_vertices);
}

template <typename T>
void Vertices<T>::set_vertex_position(int vertex_id, Vector3<T> &pos) {
    _positions(vertex_id, 0) = pos.x;
    _positions(vertex_id, 1) = pos.y;
    _positions(vertex_id, 2) = pos.z;
}

template struct Vertices<double>;
