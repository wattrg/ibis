#include <doctest/doctest.h>
#include "cell.h"
#include "interface.h"

template <typename T>
CellFaces<T>::CellFaces(const Id& interface_ids) 
{
    offsets_ = Kokkos::View<int*>("CellFaces::offsets", interface_ids.offsets().size());      
    face_ids_ = Kokkos::View<int*>("CellFaces::face_ids", interface_ids.ids().size());      
    outsigns_ = Kokkos::View<int*>("CellFaces::outsigns", interface_ids.ids().size());
    for (unsigned int i = 0; i < offsets_.size(); i++){
        offsets_(i) = interface_ids.offsets()(i);
    }
    for (unsigned int i = 0; i < face_ids_.size(); i++){
        face_ids_(i) = interface_ids.ids()(i);
        outsigns_(i) = 0;
    }
}

template <typename T>
bool CellFaces<T>::operator == (const CellFaces& other) const {
    for (unsigned int i = 0; i < offsets_.size(); i++){
        if (offsets_(i) != other.offsets_(i)) return false;
    }
    for (unsigned int i = 0; i < face_ids_.size(); i++) {
        if (face_ids_(i) != other.face_ids_(i)) return false;
    }
    for (unsigned int i = 0; i < outsigns_.size(); i++){
        if (outsigns_(i) != other.outsigns_(i)){
            for (unsigned int j = 0; j < outsigns_.size(); j++) {
                std::cout << outsigns_(j) << " ";
            }
            std::cout << std::endl;
            for (unsigned int j = 0; j < outsigns_.size(); j++) {
                std::cout << other.outsigns_(j) << " ";
            }
            std::cout << std::endl;
            return false;
        }
    }
    return true;
}

template struct CellFaces<double>;

template <typename T>
Cells<T>::Cells(Id vertices, Id interfaces, std::vector<ElemType> shapes)
        : vertex_ids_(vertices) 
{
    num_cells_ = shapes.size();
    faces_ = CellFaces<T>(interfaces);
    shape_ = Field<ElemType>("Cell::shape", num_cells_);
    for (int i = 0; i < num_cells_; i++) {
        shape_(i) = shapes[i]; 
    }

    volume_ = Field<T>("Cell::Volume", num_cells_);
    centroid_ = Vector3s<T>("Cell::centroids", num_cells_);
}

template <typename T>
void Cells<T>::compute_centroids(const Vertices<T>& vertices){
    // for the moment, we're using the arithmatic average
    // of the points as the centroid. For cells that aren't
    // nicely shaped, this could be a very bad approximation
    Kokkos::parallel_for("Cells::compute_centroid", volume_.size(), KOKKOS_LAMBDA(const int i) {
        auto cell_vertices = vertex_ids_[i];
        int n_vertices = cell_vertices.size();
        T x = 0.0;
        T y = 0.0;
        T z = 0.0;
        for (int v_idx = 0; v_idx < n_vertices; v_idx++) {
            int vertex_id = cell_vertices(v_idx);
            x += vertices.positions().x(vertex_id); 
            y += vertices.positions().y(vertex_id);
            z += vertices.positions().z(vertex_id);
        }
        centroid_.x(i) = x / n_vertices;
        centroid_.y(i) = y / n_vertices;
        centroid_.z(i) = z / n_vertices;
    });
}

template <typename T>
void Cells<T>::compute_volumes(const Vertices<T>& vertices) {
    // TODO: It would be nicer to move each case in the switch 
    // to a function sitting somewhere else to keep the amount
    // of code in this method down, and avoid duplication with
    // computing the area of interfaces. However, this won't
    // be trivial for the GPU.
    Kokkos::parallel_for("Cells::compute_volume", volume_.size(), KOKKOS_LAMBDA(const int i) {
        switch (shape_(i)) {
            case ElemType::Line:
                throw std::runtime_error("Invalid cell shape");
                break;
            case ElemType::Tri: {
                auto vertex_ids = vertex_ids_[i];
                T x1 = vertices.positions().x(vertex_ids(0));
                T x2 = vertices.positions().x(vertex_ids(1));
                T x3 = vertices.positions().x(vertex_ids(2));
                T y1 = vertices.positions().y(vertex_ids(0));
                T y2 = vertices.positions().y(vertex_ids(1));
                T y3 = vertices.positions().y(vertex_ids(2));
                T area = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2);
                volume_(i) = 0.5 * Kokkos::fabs(area);
                break;
            }
            case ElemType::Quad: {
                auto vertex_ids = vertex_ids_[i];
                T x1 = vertices.positions().x(vertex_ids(0));
                T x2 = vertices.positions().x(vertex_ids(1));
                T x3 = vertices.positions().x(vertex_ids(2));
                T x4 = vertices.positions().x(vertex_ids(3));
                T y1 = vertices.positions().y(vertex_ids(0));
                T y2 = vertices.positions().y(vertex_ids(1));
                T y3 = vertices.positions().y(vertex_ids(2));
                T y4 = vertices.positions().y(vertex_ids(3));
                T area = x1*y2 + x2*y3 + x3*y4 + x4*y1 - 
                             x2*y1 - x3*y2 - x4*y3 - x1*y4;
                volume_(i) = 0.5 * Kokkos::fabs(area);
                break;
            }
            case ElemType::Hex:
                throw std::runtime_error("Not implemented");
                break;
            case ElemType::Wedge:
                throw std::runtime_error("Not implemented");
                break;
            case ElemType::Pyramid:
                throw std::runtime_error("Not implemented");
                break;
        }
    }); 
}

template struct Cells<double>;

struct CellInfo {
    Vertices<double> vertices;
    Interfaces<double> interfaces;
    Cells<double> cells;
};

CellInfo generate_cells() {
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
    IdConstructor interface_id_constructor;
    for (unsigned int i = 0; i < interface_id_list.size(); i++){
        interface_id_constructor.push_back(interface_id_list[i]); 
    }
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
    CellInfo info;
    info.vertices = vertices;
    info.interfaces = interfaces;
    info.cells = cells;
    return info;
}

TEST_CASE("cell volume") {
    CellInfo info = generate_cells();
    Cells<double> cells = info.cells;
    Vertices<double> vertices = info.vertices;
    cells.compute_volumes(vertices);

    for (int i = 0; i < cells.size(); i++) {
        CHECK(Kokkos::fabs(cells.volume(i) - 1.0) < 1e-14);
    }
}

TEST_CASE("cell_centre") {
    CellInfo info = generate_cells();
    Cells<double> cells = info.cells;
    Vertices<double> vertices = info.vertices;

    cells.compute_centroids(vertices);

    std::vector<double> x_values = {0.5, 1.5, 2.5, 0.5, 1.5, 2.5, 0.5, 1.5, 2.5};
    std::vector<double> y_values = {0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 2.5, 2.5, 2.5};

    for (int i = 0; i < cells.size(); i++) {
        CHECK(Kokkos::fabs(cells.centroids().x(i) - x_values[i]) < 1e-14);
        CHECK(Kokkos::fabs(cells.centroids().y(i) - y_values[i]) < 1e-14);
    }
}
