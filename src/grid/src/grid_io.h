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
};

struct ElemIO {
    ElemIO(std::vector<int> ids, ElemType type) : cell_type(type), vertex_ids(ids) {}

    bool operator == (const ElemIO &other) const {
        return (vertex_ids == other.vertex_ids) && (cell_type == other.cell_type);
    }

    std::vector<int> vertex_ids {};
    ElemType cell_type;
};

struct GridIO {
public:
    GridIO(std::vector<Vertex<double>> vertices, std::vector<ElemIO> cells, 
           std::unordered_map<std::string, std::vector<ElemIO>> bcs) : 
        _vertices(vertices), _cells(cells), _bcs(bcs){}

    GridIO(std::string file_name);

    bool operator == (const GridIO &other) const {
        return (_vertices == other._vertices) && 
               (_cells == other._cells) &&
               (_bcs == other._bcs);
    }
    
private:
    std::vector<Vertex<double>> _vertices {};
    std::vector<ElemIO> _cells {};
    std::unordered_map<std::string, std::vector<ElemIO>> _bcs;

    void _read_su2_grid(std::ifstream & grid_file);
};

#endif
