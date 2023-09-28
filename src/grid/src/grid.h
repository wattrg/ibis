#ifndef GRID_H
#define GRID_H

#include <nlohmann/json.hpp>
#include "grid_io.h"
#include "../../util/src/id.h"
#include "interface.h"
#include "cell.h"

using json = nlohmann::json;

template <typename T>
class GridBlock {
public:
    GridBlock() {}

    GridBlock(const GridIO &grid_io, json boundaries);

    GridBlock(std::string file_name, json boundaries) 
        : GridBlock<T>(GridIO(file_name), boundaries) {}

    GridBlock(Vertices<T> vertices, Interfaces<T> interfaces, Cells<T> cells) 
        : vertices_(vertices), interfaces_(interfaces), cells_(cells) {}

    bool operator == (const GridBlock &other) const {
        return (vertices_ == other.vertices_) &&
               (interfaces_ == other.interfaces_) &&
               (cells_ == other.cells_);
    }

    void compute_geometric_data();

    Vertices<T>& vertices() {return vertices_;}
    const Vertices<T>& vertices() const {return vertices_;}
    int num_vertices() const {return vertices_.size();}

    Interfaces<T>& interfaces() {return interfaces_;}
    const Interfaces<T>& interfaces() const {return interfaces_;}
    int num_interfaces() const {return interfaces_.size();}

    Cells<T>& cells() {return cells_;}
    const Cells<T>& cells() const {return cells_;}
    int num_cells() const {return num_valid_cells_;}
    int num_ghost_cells() const {return num_ghost_cells_;}
    int num_total_cells() const {return num_valid_cells_+num_ghost_cells_;}

    int dim() const {return dim_;}

private:
    Vertices<T> vertices_;
    Interfaces<T> interfaces_;
    Cells<T> cells_;
    int dim_;
    int num_valid_cells_;
    int num_ghost_cells_;
    std::map<std::string, Field<int>> boundary_cells_;
    std::map<std::string, Field<int>> boundary_faces_;

    // void compute_interface_connectivity_(){
    //     interfaces_.compute_connectivity(vertices_, cells_);
    // }
};

#endif
