#ifndef GRID_H
#define GRID_H

#include "grid_io.h"
#include "../../util/src/id.h"
#include "interface.h"
#include "cell.h"

template <typename T>
class GridBlock {
public:
    GridBlock(const GridIO &grid_io);

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
        cells_.compute_centroids(vertices_);
        interfaces_.compute_areas(vertices_);
        interfaces_.compute_orientations(vertices_);
    }

    Vertices<T>& vertices() {return vertices_;}
    const Vertices<T>& vertices() const {return vertices_;}
    int num_vertices() const {return vertices_.size();}

    Interfaces<T>& interfaces() {return interfaces_;}
    const Interfaces<T>& interfaces() const {return interfaces_;}
    int num_interfaces() const {return interfaces_.size();}

    Cells<T>& cells() {return cells_;}
    const Cells<T>& cells() const {return cells_;}
    int num_cells() const {return cells_.size();}

    int dim() const {return dim_;}

private:
    Vertices<T> vertices_;
    Interfaces<T> interfaces_;
    Cells<T> cells_;
    Cells<T> ghost_;
    int dim_;

    // void compute_interface_connectivity_(){
    //     interfaces_.compute_connectivity(vertices_, cells_);
    // }
};

#endif
