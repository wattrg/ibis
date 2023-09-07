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
        dim_ = grid_io.dim();

        // set the positions of the vertices
        std::vector<Vertex<double>> vertices = grid_io.vertices();
        vertices_ = Vertices<T>(vertices.size());
        for (unsigned int i = 0; i < vertices.size(); i++) {
            vertices_.set_vertex_position(i, vertices[i].pos());
        }

        // some objects to assist in constructing the grid
        IdConstructor interface_vertices = IdConstructor();
        IdConstructor cell_vertices = IdConstructor();
        IdConstructor cell_interface_ids = IdConstructor();
        InterfaceLookup interfaces = InterfaceLookup();

        // begin to assemble the interfaces and cells
        std::vector<ElemIO> cells = grid_io.cells();
        std::vector<ElemType> cell_shapes {};
        std::vector<ElemType> interface_shapes{};
        for (unsigned int cell_i = 0; cell_i < cells.size(); cell_i++) {
            cell_vertices.push_back(cells[cell_i].vertex_ids()); 
            cell_shapes.push_back(cells[cell_i].cell_type());

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
                    interface_shapes.push_back(cell_interfaces[face_i].cell_type());
                }
                cell_face_ids.push_back(face_id);
            }
            cell_interface_ids.push_back(cell_face_ids);
        } 

        interfaces_ = Interfaces<T>(interface_vertices, interface_shapes);
        cells_ = Cells<T>(cell_vertices, cell_interface_ids, cell_shapes);

        compute_geometric_data();
    } 

    GridBlock(std::string file_name) : GridBlock<T>(GridIO(file_name)) {}

    GridBlock(Vertices<T> vertices, Interfaces<T> interfaces, Cells<T> cells) 
        : vertices_(vertices), interfaces_(interfaces), cells_(cells) {}

    bool operator == (const GridBlock &other) const {
        return (vertices_ == other.vertices_) &&
               (interfaces_ == other.interfaces_) &&
               (cells_ == other.cells_);
    }

    void compute_geometric_data() {
        cells_.compute_volumes(vertices_);
        interfaces_.compute_areas(vertices_);
        interfaces_.compute_orientations(vertices_);
    }

    Vertices<T>& vertices() {return vertices_;}
    int num_vertices() const {return vertices_.size();}

    Interfaces<T>& interfaces() {return interfaces_;}
    int num_interfaces() const {return interfaces_.size();}

    Cells<T>& cells() {return cells_;}
    int num_cells() const {return cells_.size();}


    int dim() const {return dim_;}

private:
    Vertices<T> vertices_;
    Interfaces<T> interfaces_;
    Cells<T> cells_;
    Cells<T> ghost_;
    int dim_;
};

#endif
