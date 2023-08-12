#ifndef GRID_IO_H
#define GRID_IO_H

#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <fstream>

#include "vertex.h"

enum class GridFileType {
    Native,
    Su2,
};

enum class ElemType {
    Line,
    Tri,
    Quad,
    Hex,
    Wedge,
    Pyramid,
};

enum class FaceOrder {
    Vtk,
};

struct ElemIO {
    ElemIO(std::vector<int> ids, ElemType type, FaceOrder face_order) 
        : _cell_type(type), _vertex_ids(ids), _face_order(face_order)
    {}

    bool operator == (const ElemIO &other) const {
        return (_vertex_ids == other._vertex_ids) && 
               (_cell_type == other._cell_type);
    }

    std::vector<int> vertex_ids () const {return _vertex_ids;}

    ElemType cell_type () const {return _cell_type;}

    std::vector<ElemIO> interfaces() const;

private:
    std::vector<int> _vertex_ids {};
    ElemType _cell_type;
    FaceOrder _face_order;
};

struct GridIO {
public:
    GridIO(std::vector<Vertex<double>> vertices, 
           std::vector<ElemIO> cells, 
           std::unordered_map<std::string, 
           std::vector<ElemIO>> bcs) : 
        _vertices(vertices), _cells(cells), _bcs(bcs){}

    GridIO(std::string file_name);

    bool operator == (const GridIO &other) const {
        return (_vertices == other._vertices) && 
               (_cells == other._cells) &&
               (_bcs == other._bcs);
    }

    std::vector<Vertex<double>> vertices() const {return _vertices;}

    std::vector<ElemIO> cells() const {return _cells;}

    std::unordered_map<std::string, std::vector<ElemIO>> bcs() const {
        return _bcs;
    }
    
private:
    std::vector<Vertex<double>> _vertices {};
    std::vector<ElemIO> _cells {};
    std::unordered_map<std::string, std::vector<ElemIO>> _bcs;

    void _read_su2_grid(std::ifstream & grid_file);
};

#endif
