#include "vertex.h"

Vertices::Vertices (int num_vertices) {
    _positions = Aeolus::Vector3s("vertices", num_vertices);
}

void Vertices::set_vertex_position(int vertex_id, Aeolus::Vector3 pos) {
    _positions(vertex_id, 0) = pos.x;
    _positions(vertex_id, 1) = pos.y;
    _positions(vertex_id, 2) = pos.z;
}
