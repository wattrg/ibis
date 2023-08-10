#ifndef VERTEX_H
#define VERTEX_H

#include "../../util/src/vector3.h"

struct Vertex {
public:
    Vertex(Aeolus::Vector3 pos) : _pos(pos) {}
    Aeolus::Vector3 & pos() {return _pos;}

private:
    Aeolus::Vector3 _pos;
};

struct Vertices {
public:
    Vertices(int num_vertices);
    void set_vertex_position(int vertex_id, Aeolus::Vector3 pos);
    Aeolus::Vector3s &positions() {return _positions;}

private:
    Aeolus::Vector3s _positions;
};

#endif
