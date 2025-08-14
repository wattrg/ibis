#ifndef GRID_IO_H
#define GRID_IO_H

#include <grid/vertex.h>

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

enum class GridFileType {
    Native,
    Su2,
};

enum class ElemType {
    Line,
    Tri,
    Tetra,
    Quad,
    Hex,
    Wedge,
    Pyramid,
};

enum class FaceOrder {
    Vtk,
};

struct ElemIO {
    ElemIO() {}

    ElemIO(std::vector<size_t> ids, ElemType type, FaceOrder face_order)
        : vertex_ids_(ids), cell_type_(type), face_order_(face_order) {}

    ElemIO(const ElemIO &other);

    ElemIO &operator=(const ElemIO &other);

    bool operator==(const ElemIO &other) const {
        return (vertex_ids_ == other.vertex_ids_) && (cell_type_ == other.cell_type_);
    }

    std::vector<size_t> vertex_ids() const { return vertex_ids_; }

    ElemType cell_type() const { return cell_type_; }

    FaceOrder face_order() const { return face_order_; }

    std::vector<ElemIO> interfaces() const;

    friend std::ostream &operator<<(std::ostream &file, const ElemIO &elem_io);

private:
    std::vector<size_t> vertex_ids_{};
    ElemType cell_type_;
    FaceOrder face_order_;
};


// Efficient look-up of interface ID
// from the index of the vertices
// forming the interface
struct InterfaceLookup {
public:
    InterfaceLookup();

    size_t insert(std::vector<size_t> vertex_ids);
    bool contains(std::vector<size_t> vertex_ids) const;
    size_t id(std::vector<size_t> vertex_ids) const;

private:
    std::unordered_map<std::string, size_t> hash_map_;

    std::string hash_vertex_ids(std::vector<size_t> vertex_ids) const;
    bool contains_hash(std::string hash) const;
};

struct CellMapping {
    // local information
    size_t local_cell;

    // information about the other block
    size_t other_block;
    size_t other_cell;

    // the face between the two cells
    size_t global_face;
    size_t local_face = std::numeric_limits<size_t>::max();

    CellMapping(size_t local_cell_, size_t other_block_, size_t other_cell_, size_t face_)
        : local_cell(local_cell_), other_block(other_block_),
          other_cell(other_cell_), global_face(face_) {}

    CellMapping(size_t local_cell_, size_t local_face_, size_t global_face_, size_t other_block_, size_t other_cell_)
        : local_cell(local_cell_), other_block(other_block_),
          other_cell(other_cell_), global_face(global_face_), local_face(local_face_) {}

    bool operator==(const CellMapping &other) const {
        return (local_cell == other.local_cell) && (local_face == other.local_face) &&
               (other_block == other.other_block) && (other_cell == other.other_cell) &&
                (global_face == other.global_face) && (local_face == other.local_face);
    }

    friend std::ostream &operator<<(std::ostream &file, const CellMapping &map);
};

struct GridIO {
public:
    GridIO(std::vector<Vertex<Ibis::real>> vertices, std::vector<ElemIO> cells,
           std::unordered_map<std::string, std::vector<ElemIO>> markers, int dim)
        : vertices_(vertices), cells_(cells), markers_(markers), dim_(dim) {}

    GridIO(std::vector<Vertex<Ibis::real>> vertices, std::vector<ElemIO> cells,
           std::unordered_map<std::string, std::vector<ElemIO>> markers)
        : vertices_(vertices), cells_(cells), markers_(markers) {}

    GridIO(std::string file_name);

    GridIO(const GridIO &monolithic_grid, const std::vector<size_t> &cells_to_include,
           const std::vector<CellMapping> &&cell_mapping, size_t id);

    GridIO() {}

    bool operator==(const GridIO &other) const {
        return (vertices_ == other.vertices_) && (cells_ == other.cells_) &&
               (markers_ == other.markers_) && (cell_mapping_ == other.cell_mapping_);
    }

    std::vector<Vertex<Ibis::real>> vertices() const { return vertices_; }

    std::vector<ElemIO> faces() const { return faces_; }
    const InterfaceLookup& interface_lookup() const { return interface_lookup_; }
    
    std::vector<ElemIO> cells() const { return cells_; }

    std::vector<std::vector<size_t>> cell_face_ids() const { return cell_faces_; }

    std::unordered_map<std::string, std::vector<ElemIO>> markers() const {
        return markers_;
    }

    std::vector<CellMapping> cell_mapping() const { return cell_mapping_; }

    size_t dim() const { return dim_; }

    size_t id() const { return id_; }

    void read_su2_grid(std::istream &grid_file);
    void write_su2_grid(std::ostream &grid_file);

    void read_mapped_cells(std::istream &file);
    void write_mapped_cells(std::ostream &file);

private:
    std::vector<Vertex<Ibis::real>> vertices_{};
    std::vector<ElemIO> cells_{};
    std::vector<ElemIO> faces_{};

    std::vector<std::vector<size_t>> cell_faces_; // the ID of the faces of each cell
    InterfaceLookup interface_lookup_;
    
    std::unordered_map<std::string, std::vector<ElemIO>> markers_;
    size_t dim_;
    size_t id_ = 0;

    void construct_faces_();
    

    // for partitioned grids, we need to know which cells connect
    // to cells in a different block.
    std::vector<CellMapping> cell_mapping_;
};

#endif
