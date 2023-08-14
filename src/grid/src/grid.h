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
        _vertices = Vertices<T>(vertices.size());
        for (unsigned int i = 0; i < vertices.size(); i++) {
            _vertices.set_vertex_position(i, vertices[i].pos());
        }

        // some objects to assist in constructing the grid
        IdConstructor interface_vertices = IdConstructor();
        IdConstructor cell_vertices = IdConstructor();
        IdConstructor cell_interface_ids = IdConstructor();
        InterfaceLookup interfaces = InterfaceLookup();

        // begin to assemble the interfaces and cells
        std::vector<ElemIO> cells = grid_io.cells();
        for (unsigned int cell_i = 0; cell_i < cells.size(); cell_i++) {
            cell_vertices.push_back(cells[cell_i].vertex_ids()); 

            std::vector<ElemIO> cell_interfaces = cells[cell_i].interfaces(); 
            std::vector<int> cell_face_ids {};
            for (unsigned int face_i = 0; face_i < cell_interfaces.size(); face_i++) {
                std::vector<int> face_vertices = cell_interfaces[face_i].vertex_ids();

                // if this interface already exists, we use the existing one
                // if the interface doesn't exist, we make a new one
                int face_id = interfaces.id(face_vertices);

                if (face_id == -1){
                    face_id = interfaces.insert(face_vertices);
                    interface_vertices.push_back(face_vertices);
                }
                cell_face_ids.push_back(face_id);
            }
            cell_interface_ids.push_back(cell_face_ids);
        } 

        _interfaces = Interfaces<T>(interface_vertices);
        _cells = Cells<T>(cell_vertices, cell_interface_ids);
    } 

    GridBlock(std::string file_name) : GridBlock<T>(GridIO(file_name)) {}

    GridBlock(Vertices<T> vertices, Interfaces<T> interfaces, Cells<T> cells) 
        : _vertices(vertices), _interfaces(interfaces), _cells(cells) {}

    bool operator == (const GridBlock &other) const {
        return (_vertices == other._vertices) &&
               (_interfaces == other._interfaces) &&
               (_cells == other._cells);
    }

public:
    Vertices<T> _vertices;
    Interfaces<T> _interfaces;
    Cells<T> _cells;
};

#endif
