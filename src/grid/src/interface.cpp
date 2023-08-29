#include <algorithm>
#include <functional>
#include <doctest/doctest.h>
#include "interface.h"

InterfaceLookup::InterfaceLookup() {
    _hash_map = std::unordered_map<std::string, int> {};
}

int InterfaceLookup::insert(std::vector<int> vertex_ids) {
    std::string hash = _hash(vertex_ids);
    if (_contains(hash)){
        return _hash_map[hash];
    }
    int id = _hash_map.size();
    _hash_map.insert({hash, id});
    return id;
}

bool InterfaceLookup::contains(std::vector<int> vertex_ids) {
    std::string hash = _hash(vertex_ids);
    return _contains(hash);
}

int InterfaceLookup::id(std::vector<int> vertex_ids) {
    std::string hash = _hash(vertex_ids);
    if (_contains(hash)) {
        return _hash_map[hash];
    }
    return -1;
}

bool InterfaceLookup::_contains(std::string hash) {
    return _hash_map.find(hash) != _hash_map.end();
}

std::string InterfaceLookup::_hash(std::vector<int> vertex_ids) {
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
