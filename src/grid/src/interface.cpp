#include <algorithm>
#include <functional>
#include <doctest/doctest.h>
#include "interface.h"

template <typename T>
Interfaces<T>::Interfaces(IdConstructor ids, std::vector<ElemType> shapes)
    : m_vertex_ids(Id(ids))
{
    size_ = m_vertex_ids.size();
    shape_ = Field<ElemType>("Interface::shape", shapes.size());
    for (int i = 0; i < size_; i++) {
        shape_(i) = shapes[i];
    }

    // geometry
    norm_ = Vector3s<T>("Interface::norm", size_);
    tan1_ = Vector3s<T>("Interface::tan1", size_);
    tan2_ = Vector3s<T>("Interface::tan2", size_);
    area_ = Field<T>("Interface::area", size_);
    centre_ = Vector3s<T>("Interface::centre", size_);

    // set left and right cells to -1 to indicate they haven't
    // been connected up to any cells yet
    left_cells_ = Field<int>("Interface::left", size_);
    right_cells_ = Field<int>("Interface::right", size_);
    for (int i = 0; i < size_; i++) {
        left_cells_(i) = -1;
        right_cells_(i) = -1;
    }

    // on_boundary_ = Field<bool>("Interface::on_boundary", size_);
}

template <typename T>
void Interfaces<T>::compute_orientations(Vertices<T> vertices) {
    // set the face tangents in parallel
    Kokkos::parallel_for("Interfaces::compute_orientations", norm_.size(), KOKKOS_LAMBDA (const int i){
        auto vertex_ids = m_vertex_ids[i];
        T x0 = vertices.position(vertex_ids(0), 0);
        T x1 = vertices.position(vertex_ids(1), 0);
        T y0 = vertices.position(vertex_ids(0), 1);
        T y1 = vertices.position(vertex_ids(1), 1);
        T z0 = vertices.position(vertex_ids(0), 2);
        T z1 = vertices.position(vertex_ids(1), 2);
        T ilength = 1./Kokkos::sqrt((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0) + (z1-z0));
        tan1_(i, 0) = ilength * (x1 - x0);
        tan1_(i, 1) = ilength * (y1 - y0);
        tan1_(i, 2) = ilength * (z1 - z0);

        switch (shape_(i)) {
            case ElemType::Line: {
                auto vertex_ids = m_vertex_ids[i];
                tan2_(i, 0) = 0.0;
                tan2_(i, 1) = 0.0;
                tan2_(i, 2) = 1.0;
                break;
            }
            case ElemType::Tri: 
                throw std::runtime_error("Not implemented"); 
            case ElemType::Quad:
                throw std::runtime_error("Not implemented");
            default:
                throw std::runtime_error("Invalid interface");
        }
    });

    // the face normal is the cross product of the tangents
    cross(tan1_, tan2_, norm_);
}

template <typename T>
void Interfaces<T>::compute_areas(Vertices<T> vertices) {
    Kokkos::parallel_for("Interfaces::compute_areas", area_.size(), KOKKOS_LAMBDA (const int i) {
        switch (shape_(i)) {
            case ElemType::Line: {
                auto vertex_ids = m_vertex_ids[i];
                T x1 = vertices.position(vertex_ids(0), 0);
                T x2 = vertices.position(vertex_ids(1), 0);
                T y1 = vertices.position(vertex_ids(0), 1);
                T y2 = vertices.position(vertex_ids(1), 1);
                area_(i) = Kokkos::sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1));
                break;
            }
            case ElemType::Tri: {
                auto vertex_ids = m_vertex_ids[i];
                T x1 = vertices.position(vertex_ids(0), 0);
                T x2 = vertices.position(vertex_ids(1), 0);
                T x3 = vertices.position(vertex_ids(2), 0);
                T y1 = vertices.position(vertex_ids(0), 1);
                T y2 = vertices.position(vertex_ids(1), 1);
                T y3 = vertices.position(vertex_ids(2), 1);
                T area = 0.5*Kokkos::fabs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2));
                area_(i) = area;
                break;
            }
            case ElemType::Quad: {
                auto vertex_ids = m_vertex_ids[i];
                T x1 = vertices.position(vertex_ids(0), 0);
                T x2 = vertices.position(vertex_ids(1), 0);
                T x3 = vertices.position(vertex_ids(2), 0);
                T x4 = vertices.position(vertex_ids(3), 0);
                T y1 = vertices.position(vertex_ids(0), 1);
                T y2 = vertices.position(vertex_ids(1), 1);
                T y3 = vertices.position(vertex_ids(2), 1);
                T y4 = vertices.position(vertex_ids(3), 1);
                T area = x1*y2 + x2*y3 + x3*y4 + x4*y1 - x2*y1 - x3*y2 - x4*y3 - x1*y4;
                area_(i) = 0.5 * Kokkos::fabs(area);
                break;
            }
            case ElemType::Hex: {
                throw std::runtime_error("Invalid interface"); 
            }
            case ElemType::Wedge: {
                throw std::runtime_error("Invalid interface"); 
            }
            case ElemType::Pyramid: {
                throw std::runtime_error("Invalid interface"); 
            }
        }
    });
}

template <typename T>
void Interfaces<T>::compute_centres(Vertices<T> vertices){
    Kokkos::parallel_for("Interfaces::compute_centres", centre_.size(), KOKKOS_LAMBDA (const int face_i){
        auto face_vertices = m_vertex_ids[face_i]; 
        T x = 0.0;
        T y = 0.0;
        T z = 0.0;
        for (unsigned int vtx_i = 0; vtx_i < face_vertices.size(); vtx_i++) {
            int vtx_id = face_vertices[vtx_i];
            x += vertices.positions().x(vtx_id);  
            y += vertices.positions().y(vtx_id);
            z += vertices.positions().z(vtx_id);
        }
        centre_.x(face_i) = x;
        centre_.y(face_i) = y;
        centre_.z(face_i) = z;
    });
}

template struct Interfaces<double>;

InterfaceLookup::InterfaceLookup() {
    hash_map_ = std::unordered_map<std::string, int> {};
}

int InterfaceLookup::insert(std::vector<int> vertex_ids) {
    std::string hash = hash_vertex_ids(vertex_ids);
    if (contains_hash(hash)){
        return hash_map_[hash];
    }
    int id = hash_map_.size();
    hash_map_.insert({hash, id});
    return id;
}

bool InterfaceLookup::contains(std::vector<int> vertex_ids) {
    std::string hash = hash_vertex_ids(vertex_ids);
    return contains_hash(hash);
}

int InterfaceLookup::id(std::vector<int> vertex_ids) {
    std::string hash = hash_vertex_ids(vertex_ids);
    if (contains_hash(hash)) {
        return hash_map_[hash];
    }
    return -1;
}

bool InterfaceLookup::contains_hash(std::string hash) {
    return hash_map_.find(hash) != hash_map_.end();
}

std::string InterfaceLookup::hash_vertex_ids(std::vector<int> vertex_ids) {
    std::sort(vertex_ids.begin(), vertex_ids.end(), std::greater<int>());
    std::string hash_value = "";
    for (unsigned int i = 0; i < vertex_ids.size(); i++) {
        hash_value.append(std::to_string(vertex_ids[i]));
        hash_value.append(",");
    }
    return hash_value;
}

TEST_CASE("interface look up contains") {
    InterfaceLookup x;
    x.insert(std::vector<int> {0, 1});
    x.insert(std::vector<int> {1, 5});
    x.insert(std::vector<int> {5, 4});
    x.insert(std::vector<int> {5, 1});

    CHECK(x.contains(std::vector<int> {1, 0}));
    CHECK(x.contains(std::vector<int> {6, 1}) == false);
}

TEST_CASE("interface look up id 1") {
    InterfaceLookup x;
    x.insert(std::vector<int> {0, 1});
    x.insert(std::vector<int> {1, 5});
    x.insert(std::vector<int> {5, 4});
    x.insert(std::vector<int> {5, 1});

    CHECK(x.id(std::vector<int> {5, 1}) == 1);
    CHECK(x.id(std::vector<int> {1, 5}) == 1);
}

TEST_CASE("interface look up id 2") {
    InterfaceLookup x;
    x.insert(std::vector<int> {0, 1});
    x.insert(std::vector<int> {1, 5});
    x.insert(std::vector<int> {5, 4});
    x.insert(std::vector<int> {5, 1});

    CHECK(x.id(std::vector<int> {6, 1}) == -1);
}

TEST_CASE("interface look up") {
    InterfaceLookup x;
    x.insert(std::vector<int> {0, 1});
    x.insert(std::vector<int> {1, 5});
    x.insert(std::vector<int> {5, 4});
    x.insert(std::vector<int> {4, 0});
    x.insert(std::vector<int> {1, 2});
    x.insert(std::vector<int> {2, 6});
    x.insert(std::vector<int> {6, 5});
    x.insert(std::vector<int> {5, 1});
    x.insert(std::vector<int> {2, 3});
    x.insert(std::vector<int> {3, 7});
    x.insert(std::vector<int> {7, 6});
    x.insert(std::vector<int> {6, 2});

    CHECK(x.id(std::vector<int> {0, 1}) == 0);
    CHECK(x.id(std::vector<int> {1, 5}) == 1);
    CHECK(x.id(std::vector<int> {5, 4}) == 2);
    CHECK(x.id(std::vector<int> {4, 0}) == 3);
    CHECK(x.id(std::vector<int> {1, 2}) == 4);
    CHECK(x.id(std::vector<int> {2, 6}) == 5);
    CHECK(x.id(std::vector<int> {6, 5}) == 6);
    CHECK(x.id(std::vector<int> {2, 3}) == 7);
    CHECK(x.id(std::vector<int> {3, 7}) == 8);
    CHECK(x.id(std::vector<int> {7, 6}) == 9);
}

TEST_CASE("Interface Geometry") {
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
    interfaces.compute_areas(vertices);
    interfaces.compute_orientations(vertices);

    for (unsigned int i = 0; i < shapes.size(); i++){
        CHECK(Kokkos::abs(interfaces.area(i) - 1.0) < 1e-14);
    }

    CHECK(Kokkos::abs(interfaces.norm(0).x() - +0.0) < 1e-14);
    CHECK(Kokkos::abs(interfaces.norm(0).y() - -1.0) < 1e-14);
    CHECK(Kokkos::abs(interfaces.norm(0).z() - +0.0) < 1e-14);

    CHECK(Kokkos::abs(interfaces.norm(1).x() - +1.0) < 1e-14);
    CHECK(Kokkos::abs(interfaces.norm(1).y() - +0.0) < 1e-14);
    CHECK(Kokkos::abs(interfaces.norm(1).z() - +0.0) < 1e-14);

    CHECK(Kokkos::abs(interfaces.norm(2).x() - +0.0) < 1e-14);
    CHECK(Kokkos::abs(interfaces.norm(2).y() - +1.0) < 1e-14);
    CHECK(Kokkos::abs(interfaces.norm(2).z() - +0.0) < 1e-14);

    CHECK(Kokkos::abs(interfaces.norm(3).x() - -1.0) < 1e-14);
    CHECK(Kokkos::abs(interfaces.norm(3).y() - +0.0) < 1e-14);
    CHECK(Kokkos::abs(interfaces.norm(3).z() - +0.0) < 1e-14);

    CHECK(Kokkos::abs(interfaces.norm(4).x() - +0.0) < 1e-14);
    CHECK(Kokkos::abs(interfaces.norm(4).y() - -1.0) < 1e-14);
    CHECK(Kokkos::abs(interfaces.norm(4).z() - +0.0) < 1e-14);

    CHECK(Kokkos::abs(interfaces.norm(5).x() - +1.0) < 1e-14);
    CHECK(Kokkos::abs(interfaces.norm(5).y() - +0.0) < 1e-14);
    CHECK(Kokkos::abs(interfaces.norm(5).z() - +0.0) < 1e-14);
}
