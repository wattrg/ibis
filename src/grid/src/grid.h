#ifndef GRID_H
#define GRID_H

#include <nlohmann/json.hpp>
#include "Kokkos_Core_fwd.hpp"
#include "grid_io.h"
#include "../../util/src/id.h"
#include "interface.h"
#include "cell.h"

using json = nlohmann::json;

template <typename T,
          class ExecSpace=Kokkos::DefaultExecutionSpace,
          class Layout=Kokkos::DefaultExecutionSpace::array_layout>
class GridBlock {
public:
    using execution_space = ExecSpace;
    using memory_space = typename execution_space::memory_space;
    using array_layout = Layout;
    using host_execution_space = Kokkos::DefaultHostExecutionSpace;
    using host_mirror_mem_space = Kokkos::DefaultHostExecutionSpace::memory_space;
    using mirror_type = Cells<T, Kokkos::DefaultHostExecutionSpace, array_layout>;

public:
    GridBlock() {}

    GridBlock(const GridIO &grid_io, json &config){
        dim_ = grid_io.dim();
        json boundaries = config.at("boundaries");

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
        num_valid_cells_ = cells.size();
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

        std::map<int, int> ghost_cell_map = setup_boundaries(
            grid_io, boundaries, cell_vertices, interfaces, cell_shapes
        );

        interfaces_ = Interfaces<T>(interface_vertices, interface_shapes);
        cells_ = Cells<T>(cell_vertices, cell_interface_ids, cell_shapes);

        compute_geometric_data();
        compute_interface_connectivity(ghost_cell_map);

    }

    GridBlock(std::string file_name, json &config) 
        : GridBlock<T>(GridIO(file_name), config) {}

    GridBlock(Vertices<T, execution_space, array_layout> vertices, 
              Interfaces<T, execution_space, array_layout> interfaces, 
              Cells<T, execution_space, array_layout> cells) 
        : vertices_(vertices), interfaces_(interfaces), cells_(cells) {}


    bool operator == (const GridBlock &other) const {
        return (vertices_ == other.vertices_) &&
               (interfaces_ == other.interfaces_) &&
               (cells_ == other.cells_);
    }

    void compute_geometric_data(){
        cells_.compute_centroids(vertices_);
        cells_.compute_volumes(vertices_);
        interfaces_.compute_centres(vertices_);
        interfaces_.compute_areas(vertices_);
        interfaces_.compute_orientations(vertices_);
    }

    KOKKOS_INLINE_FUNCTION
    Vertices<T, execution_space, array_layout>& vertices() {return vertices_;}
    
    KOKKOS_INLINE_FUNCTION
    const Vertices<T, execution_space, array_layout>& vertices() const {return vertices_;}

    int num_vertices() const {return vertices_.size();}

    KOKKOS_INLINE_FUNCTION
    Interfaces<T, execution_space, array_layout>& interfaces() {return interfaces_;}

    KOKKOS_INLINE_FUNCTION
    const Interfaces<T, execution_space, array_layout>& interfaces() const {return interfaces_;}

    int num_interfaces() const {return interfaces_.size();}

    KOKKOS_INLINE_FUNCTION
    Cells<T, execution_space, array_layout>& cells() {return cells_;}

    KOKKOS_INLINE_FUNCTION
    const Cells<T, execution_space, array_layout>& cells() const {return cells_;}

    int num_cells() const {return num_valid_cells_;}
    int num_ghost_cells() const {return num_ghost_cells_;}
    int num_total_cells() const {return num_valid_cells_+num_ghost_cells_;}

    KOKKOS_INLINE_FUNCTION
    bool is_valid(const int i) const {return i < num_valid_cells_;}

    const Field<int, array_layout, memory_space>& boundary_faces(std::string boundary_tag) const{
        return boundary_faces_.at(boundary_tag);
    }

    const std::vector<std::string>& boundary_tags() const {
        return boundary_tags_;
    }

    int dim() const {return dim_;}

    void compute_interface_connectivity(std::map<int, int> ghost_cells){
        auto this_interfaces = interfaces_;
        auto this_cells = cells_;
        Kokkos::parallel_for("compute_interface_connectivity", 
                             Kokkos::RangePolicy<execution_space>(0, num_valid_cells_),
                             KOKKOS_LAMBDA (const int cell_i){
            auto face_ids = this_cells.faces().face_ids(cell_i);
            T cell_x = this_cells.centroids().x(cell_i);
            T cell_y = this_cells.centroids().y(cell_i);
            T cell_z = this_cells.centroids().z(cell_i);
            for (unsigned int face_i = 0; face_i < face_ids.size(); face_i++){
                int face_id = face_ids[face_i];
                
                // vector from the face centre to the cell centre 
                T dx = this_interfaces.centre().x(face_id) - cell_x;
                T dy = this_interfaces.centre().y(face_id) - cell_y;
                T dz = this_interfaces.centre().z(face_id) - cell_z;

                // dot product of the vector from centre to centre with
                // the interface normal vector
                T dot = dx * this_interfaces.norm().x(face_id) +
                        dy * this_interfaces.norm().y(face_id) +
                        dz * this_interfaces.norm().z(face_id);
                if (dot > 0.0) {
                    // cell is on the left of the face
                    this_interfaces.attach_cell_left(cell_i, face_id);
                    this_cells.faces().set_outsign(cell_i, face_i, 1);
                }
                else {
                    // cell is on the right of face
                    this_interfaces.attach_cell_right(cell_i, face_id);
                    this_cells.faces().set_outsign(cell_i, face_i, -1);
                }
            }
        });

        // TODO: loop through the ghost cells and attach them to
        // the other side of the interface.
        // Think about how to do this on the GPU
        for (auto boundary : ghost_cells){
            int face_id = boundary.first;
            int ghost_cell_id = boundary.second;
            if (interfaces_.left_cell(face_id) < 0) {
                interfaces_.attach_cell_left(ghost_cell_id, face_id);
            } 
            else {
                interfaces_.attach_cell_right(ghost_cell_id, face_id);
            }
        }
    }

public:
    Vertices<T, execution_space, array_layout> vertices_;
    Interfaces<T, execution_space, array_layout> interfaces_;
    Cells<T, execution_space, array_layout> cells_;
    int dim_;
    int num_valid_cells_;
    int num_ghost_cells_;
    std::map<std::string, Field<int>> boundary_cells_;
    std::map<std::string, Field<int>> boundary_faces_;
    std::vector<std::string> boundary_tags_;

    std::map<int, int> setup_boundaries(const GridIO & grid_io, json& boundaries,
                                        IdConstructor &cell_vertices, 
                                        InterfaceLookup& interfaces,
                                        std::vector<ElemType> cell_shapes){
        (void) cell_vertices;
        (void) cell_shapes;
        num_ghost_cells_ = 0;
        std::map<int, int> ghost_cell_map; // face_id -> ghost_cell_id
        for (auto bc : grid_io.bcs()) {
            // unpack the boundary data from the grid_io object
            std::string bc_label = bc.first;
            boundary_tags_.push_back(bc_label);
            std::vector<ElemIO> bc_faces = bc.second;
            json boundary_config = boundaries.at(bc_label);

            // loop over all the boundary faces for this boundary, keeping track of 
            // which ones belong to this boundary
            std::vector<int> boundary_cells{};
            std::vector<int> boundary_faces{};
            for (unsigned int boundary_i = 0; boundary_i < bc_faces.size(); boundary_i++){
                int face_id = interfaces.id(bc_faces[boundary_i].vertex_ids());
                boundary_faces.push_back(face_id);
                if (boundary_config.at("ghost_cells") == true) {
                    // currently we're not going to actually build a 'cell',
                    // just 
                    // cell_vertices.push_back({-1, -1});
                    // cell_shapes.push_back(ElemType::Line);
                    int ghost_cell_id = num_valid_cells_ + num_ghost_cells_;
                    num_ghost_cells_++;
                    boundary_cells.push_back(ghost_cell_id);
                    ghost_cell_map.insert({face_id, ghost_cell_id});
                }
                else {
                    ghost_cell_map.insert({face_id, -1}); // no ghost cell
                }
            }

            // keep track of which faces/cells belong to
            // which boundary
            boundary_cells_.insert(
                {bc_label, Field<int>("bc_cells", boundary_cells)}
            );
            boundary_faces_.insert(
                {bc_label, Field<int>("bc_faces", boundary_faces)}
            );
        }
        return ghost_cell_map;
    }

};

#endif
