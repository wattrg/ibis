#include <doctest/doctest.h>
#include "grid.h"
#include "grid_io.h"
#include "interface.h"

template <typename T>
GridBlock<T>::GridBlock(const GridIO& grid_io, json& config){
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
    compute_interface_connectivity_(ghost_cell_map);
} 

template <typename T>
void GridBlock<T>::compute_geometric_data() {
    cells_.compute_volumes(vertices_);
    cells_.compute_centroids(vertices_);
    interfaces_.compute_areas(vertices_);
    interfaces_.compute_orientations(vertices_);
    interfaces_.compute_centres(vertices_);
}

template <typename T>
std::map<int, int> GridBlock<T>::setup_boundaries(const GridIO& grid_io, 
                                    json& boundaries,
                                    IdConstructor &cell_vertices, 
                                    InterfaceLookup& interfaces,
                                    std::vector<ElemType> cell_shapes) {
    (void) cell_vertices;
    (void) cell_shapes;
    num_ghost_cells_ = 0;
    std::map<int, int> ghost_cell_map; // face_id -> ghost_cell_id
    for (auto bc : grid_io.bcs()) {
        // unpack the boundary data from the grid_io object
        std::string bc_label = bc.first;
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
                num_ghost_cells_++;
                int ghost_cell_id = num_valid_cells_ + num_ghost_cells_;
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

template <typename T> 
void GridBlock<T>::compute_interface_connectivity_(std::map<int, int> ghost_cells) {
    Kokkos::parallel_for("compute_interface_connectivity", num_valid_cells_, KOKKOS_LAMBDA (const int cell_i){
        auto face_ids = cells_.interface_ids()[cell_i];
        T cell_x = cells_.centroids().x(cell_i);
        T cell_y = cells_.centroids().y(cell_i);
        T cell_z = cells_.centroids().z(cell_i);
        for (unsigned int face_i = 0; face_i < face_ids.size(); face_i++){
            int face_id = face_ids[face_i];
            
            // vector from the face centre to the cell centre 
            T dx = interfaces_.centre().x(face_id) - cell_x;
            T dy = interfaces_.centre().y(face_id) - cell_y;
            T dz = interfaces_.centre().z(face_id) - cell_z;

            // dot product of the vector from centre to centre with
            // the interface normal vector
            T dot = dx * interfaces_.norm().x(face_id) +
                    dy * interfaces_.norm().y(face_id) +
                    dz * interfaces_.norm().z(face_id);
            if (dot > 0.0) {
                // cell is on the left of the face
                interfaces_.attach_cell_left(cell_i, face_id);
                cells_.set_outsign(cell_i, face_i, 1);
            }
            else {
                // cell is on the right of face
                interfaces_.attach_cell_right(cell_i, face_id);
                cells_.set_outsign(cell_i, face_i, -1);
            }
        }
    });

    // TODO: loop through the ghost cells and attach them to
    // the other side of the interface
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

template class GridBlock<double>;

TEST_CASE("build grid block") {
    Vertices<double> vertices(16);
    std::vector<Vector3<double>> vertex_pos {
        Vector3<double>(0.0, 0.0, 0.0),
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(2.0, 0.0, 0.0),
        Vector3<double>(3.0, 0.0, 0.0),
        Vector3<double>(0.0, 1.0, 0.0),
        Vector3<double>(1.0, 1.0, 0.0),
        Vector3<double>(2.0, 1.0, 0.0),
        Vector3<double>(3.0, 1.0, 0.0),
        Vector3<double>(0.0, 2.0, 0.0),
        Vector3<double>(1.0, 2.0, 0.0),
        Vector3<double>(2.0, 2.0, 0.0),
        Vector3<double>(3.0, 2.0, 0.0),
        Vector3<double>(0.0, 3.0, 0.0),
        Vector3<double>(1.0, 3.0, 0.0),
        Vector3<double>(2.0, 3.0, 0.0),
        Vector3<double>(3.0, 3.0, 0.0)
    };
    for (int i = 0; i < 16; i++) {
        vertices.set_vertex_position(i, vertex_pos[i]);
    }

    std::vector<std::vector<int>> interface_id_list {
        {0, 1},
        {1, 5},
        {5, 4},
        {4, 0},
        {1, 2},
        {2, 6},
        {6, 5},
        {2, 3},
        {3, 7},
        {7, 6},
        {5, 9},
        {9, 8},
        {8, 4},
        {6, 10},
        {10, 9},
        {7, 11},
        {11, 10},
        {9, 13},
        {13, 12},
        {12, 8},
        {10, 14},
        {14, 13},
        {11, 15},
        {15, 14}
    };
    std::vector<ElemType> shapes = {
        ElemType::Line,
        ElemType::Line,
        ElemType::Line,
        ElemType::Line,
        ElemType::Line,
        ElemType::Line,
        ElemType::Line,
        ElemType::Line,
        ElemType::Line,
        ElemType::Line,
        ElemType::Line,
        ElemType::Line,
        ElemType::Line,
        ElemType::Line,
        ElemType::Line,
        ElemType::Line,
        ElemType::Line,
        ElemType::Line,
        ElemType::Line,
        ElemType::Line,
        ElemType::Line,
        ElemType::Line,
        ElemType::Line,
        ElemType::Line,
    };
    IdConstructor interface_id_constructor;
    for (unsigned int i = 0; i < interface_id_list.size(); i++){
        interface_id_constructor.push_back(interface_id_list[i]); 
    }
    Interfaces<double> interfaces (interface_id_constructor, shapes);

    std::vector<std::vector<int>> cell_interfaces_list {
        {0, 1, 2, 3},
        {4, 5, 6, 1},
        {7, 8, 9, 5},
        {2, 10, 11, 12},
        {6, 13, 14, 10},
        {9, 15, 16, 13},
        {11, 17, 18, 19},
        {14, 20, 21, 17},
        {16, 22, 23, 20}
    };
    IdConstructor cell_interface_id_constructor;
    for (unsigned int i = 0; i < cell_interfaces_list.size(); i++) {
        cell_interface_id_constructor.push_back(cell_interfaces_list[i]);
    }

    IdConstructor cell_vertex_id_constructor;
    std::vector<std::vector<int>> cell_vertex_ids_raw {
        {0, 1, 5, 4},
        {1, 2, 6, 5},
        {2, 3, 7, 6},
        {4, 5, 9, 8},
        {5, 6, 10, 9},
        {6, 7, 11, 10},
        {8, 9, 13, 12},
        {9, 10, 14, 13},
        {10, 11, 15, 14}
    };
    for (int i = 0; i < 9; i++) {
        cell_vertex_id_constructor.push_back(cell_vertex_ids_raw[i]);
    }
    std::vector<ElemType> cell_shapes {
        ElemType::Quad,
        ElemType::Quad,
        ElemType::Quad,
        ElemType::Quad,
        ElemType::Quad,
        ElemType::Quad,
        ElemType::Quad,
        ElemType::Quad,
        ElemType::Quad,
    };

    Cells<double> cells (cell_vertex_id_constructor, cell_interface_id_constructor, cell_shapes);

    GridBlock<double> expected = GridBlock<double>(vertices, interfaces, cells);
    json config{};
    json boundaries {};
    json slip_wall{};
    json inflow{};
    json outflow{};
    slip_wall["ghost_cells"] = false;
    inflow["ghost_cells"] = false;
    outflow["ghost_cells"] = false;
    boundaries["slip_wall_bottom"] = slip_wall;
    boundaries["slip_wall_top"] = slip_wall;
    boundaries["inflow"] = inflow;
    boundaries["outflow"] = outflow;
    config["boundaries"] = boundaries;
    GridBlock<double> block = GridBlock<double>("../src/grid/test/grid.su2", config);
    CHECK(block == expected);
}
