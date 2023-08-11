#ifndef GRID_H
#define GRID_H

#include "grid_io.h"
#include "../../util/src/id.h"
#include "interface.h"
#include "cell.h"

template <typename T>
struct GridBlock {
public:
    GridBlock(const GridIO &grid_io){
        // set the positions of the vertices
        std::vector<Vertex<double>> vertices = grid_io.vertices();
        for (int i = 0; i < vertices.size(); i++) {
            _vertices.set_vertex_position(i, vertices[i]);
        }

        // some objects to assist in constructing the grid
        IdConstructor interface_vertices = IdConstructor();
        IdConstructor cell_vertices = IdConstructor();
        IdConstructor cell_interface = IdConstructor();
        InterfaceLookup interfaces = InterfaceLookup();

        // begin to assemble the interfaces and cells
        std::vector<ElemIO> cells = grid_io.cells();
        for (int i = 0; i < cells.size(); i++) {
            cell_vertices.push_back(cells[i].vertex_ids()); 

            // need to write ElemIO::interfaces
        } 
    } 

private:
    Vertices<T> _vertices;
    Interfaces<T> _interfaces;
    Cells<T> _cells;
};

#endif
