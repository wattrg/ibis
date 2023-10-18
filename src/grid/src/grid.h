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

    GridBlock(const GridIO &grid_io, json &config);

    GridBlock(std::string file_name, json &config) 
        : GridBlock<T>(GridIO(file_name), config) {}

    GridBlock(Vertices<T> vertices, Interfaces<T> interfaces, Cells<T> cells) 
        : vertices_(vertices), interfaces_(interfaces), cells_(cells) {}


    bool operator == (const GridBlock &other) const {
        return (vertices_ == other.vertices_) &&
               (interfaces_ == other.interfaces_) &&
               (cells_ == other.cells_);
    }

    void compute_geometric_data();

    KOKKOS_INLINE_FUNCTION
    Vertices<T>& vertices() {return vertices_;}
    
    KOKKOS_INLINE_FUNCTION
    const Vertices<T>& vertices() const {return vertices_;}

    int num_vertices() const {return vertices_.size();}

    KOKKOS_INLINE_FUNCTION
    Interfaces<T>& interfaces() {return interfaces_;}

    KOKKOS_INLINE_FUNCTION
    const Interfaces<T>& interfaces() const {return interfaces_;}

    int num_interfaces() const {return interfaces_.size();}

    KOKKOS_INLINE_FUNCTION
    Cells<T>& cells() {return cells_;}

    KOKKOS_INLINE_FUNCTION
    const Cells<T>& cells() const {return cells_;}

    int num_cells() const {return num_valid_cells_;}
    int num_ghost_cells() const {return num_ghost_cells_;}
    int num_total_cells() const {return num_valid_cells_+num_ghost_cells_;}

    KOKKOS_INLINE_FUNCTION
    bool is_valid(const int i) const {return i < num_valid_cells_;}

    const Field<int>& boundary_faces(std::string boundary_tag) const;
    const std::vector<std::string>& boundary_tags() const;

    int dim() const {return dim_;}

    void compute_interface_connectivity(std::map<int, int> ghost_cells);

private:
    Vertices<T> vertices_;
    Interfaces<T> interfaces_;
    Cells<T> cells_;
    int dim_;
    int num_valid_cells_;
    int num_ghost_cells_;
    std::map<std::string, Field<int>> boundary_cells_;
    std::map<std::string, Field<int>> boundary_faces_;
    std::vector<std::string> boundary_tags_;

    std::map<int, int> setup_boundaries(
        const GridIO & grid_io, json& boundaries,
        IdConstructor &cell_vertices, 
        InterfaceLookup& interfaces,
        std::vector<ElemType> cell_shapes);

};

#endif
