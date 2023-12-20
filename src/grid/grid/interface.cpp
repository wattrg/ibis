#include <doctest/doctest.h>
#include <grid/interface.h>

#include <algorithm>
#include <functional>

InterfaceLookup::InterfaceLookup() {
    hash_map_ = std::unordered_map<std::string, int>{};
}

int InterfaceLookup::insert(std::vector<int> vertex_ids) {
    std::string hash = hash_vertex_ids(vertex_ids);
    if (contains_hash(hash)) {
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
    x.insert(std::vector<int>{0, 1});
    x.insert(std::vector<int>{1, 5});
    x.insert(std::vector<int>{5, 4});
    x.insert(std::vector<int>{5, 1});

    CHECK(x.contains(std::vector<int>{1, 0}));
    CHECK(x.contains(std::vector<int>{6, 1}) == false);
}

TEST_CASE("interface look up id 1") {
    InterfaceLookup x;
    x.insert(std::vector<int>{0, 1});
    x.insert(std::vector<int>{1, 5});
    x.insert(std::vector<int>{5, 4});
    x.insert(std::vector<int>{5, 1});

    CHECK(x.id(std::vector<int>{5, 1}) == 1);
    CHECK(x.id(std::vector<int>{1, 5}) == 1);
}

TEST_CASE("interface look up id 2") {
    InterfaceLookup x;
    x.insert(std::vector<int>{0, 1});
    x.insert(std::vector<int>{1, 5});
    x.insert(std::vector<int>{5, 4});
    x.insert(std::vector<int>{5, 1});

    CHECK(x.id(std::vector<int>{6, 1}) == -1);
}

TEST_CASE("interface look up") {
    InterfaceLookup x;
    x.insert(std::vector<int>{0, 1});
    x.insert(std::vector<int>{1, 5});
    x.insert(std::vector<int>{5, 4});
    x.insert(std::vector<int>{4, 0});
    x.insert(std::vector<int>{1, 2});
    x.insert(std::vector<int>{2, 6});
    x.insert(std::vector<int>{6, 5});
    x.insert(std::vector<int>{5, 1});
    x.insert(std::vector<int>{2, 3});
    x.insert(std::vector<int>{3, 7});
    x.insert(std::vector<int>{7, 6});
    x.insert(std::vector<int>{6, 2});

    CHECK(x.id(std::vector<int>{0, 1}) == 0);
    CHECK(x.id(std::vector<int>{1, 5}) == 1);
    CHECK(x.id(std::vector<int>{5, 4}) == 2);
    CHECK(x.id(std::vector<int>{4, 0}) == 3);
    CHECK(x.id(std::vector<int>{1, 2}) == 4);
    CHECK(x.id(std::vector<int>{2, 6}) == 5);
    CHECK(x.id(std::vector<int>{6, 5}) == 6);
    CHECK(x.id(std::vector<int>{2, 3}) == 7);
    CHECK(x.id(std::vector<int>{3, 7}) == 8);
    CHECK(x.id(std::vector<int>{7, 6}) == 9);
}

Interfaces<double> generate_interfaces() {
    Vertices<double> vertices(16);
    auto vertices_host = vertices.host_mirror();
    std::vector<Vector3<double>> vertex_pos{
        Vector3<double>(0.0, 0.0, 0.0), Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(2.0, 0.0, 0.0), Vector3<double>(3.0, 0.0, 0.0),
        Vector3<double>(0.0, 1.0, 0.0), Vector3<double>(1.0, 1.0, 0.0),
        Vector3<double>(2.0, 1.0, 0.0), Vector3<double>(3.0, 1.0, 0.0),
        Vector3<double>(0.0, 2.0, 0.0), Vector3<double>(1.0, 2.0, 0.0),
        Vector3<double>(2.0, 2.0, 0.0), Vector3<double>(3.0, 2.0, 0.0),
        Vector3<double>(0.0, 3.0, 0.0), Vector3<double>(1.0, 3.0, 0.0),
        Vector3<double>(2.0, 3.0, 0.0), Vector3<double>(3.0, 3.0, 0.0)};
    for (int i = 0; i < 16; i++) {
        vertices_host.set_vertex_position(i, vertex_pos[i]);
    }
    vertices.deep_copy(vertices_host);

    std::vector<std::vector<int>> interface_id_list{
        {0, 1},   {1, 5},  {5, 4},   {4, 0},   {1, 2},   {2, 6},
        {6, 5},   {2, 3},  {3, 7},   {7, 6},   {5, 9},   {9, 8},
        {8, 4},   {6, 10}, {10, 9},  {7, 11},  {11, 10}, {9, 13},
        {13, 12}, {12, 8}, {10, 14}, {14, 13}, {11, 15}, {15, 14}};
    std::vector<ElemType> shapes = {
        ElemType::Line, ElemType::Line, ElemType::Line, ElemType::Line,
        ElemType::Line, ElemType::Line, ElemType::Line, ElemType::Line,
        ElemType::Line, ElemType::Line, ElemType::Line, ElemType::Line,
        ElemType::Line, ElemType::Line, ElemType::Line, ElemType::Line,
        ElemType::Line, ElemType::Line, ElemType::Line, ElemType::Line,
        ElemType::Line, ElemType::Line, ElemType::Line, ElemType::Line,
    };
    IdConstructor interface_id_constructor;
    for (unsigned int i = 0; i < interface_id_list.size(); i++) {
        interface_id_constructor.push_back(interface_id_list[i]);
    }
    Interfaces<double> interfaces(interface_id_constructor, shapes);
    interfaces.compute_areas(vertices);
    interfaces.compute_orientations(vertices);
    interfaces.compute_centres(vertices);
    return interfaces;
}

TEST_CASE("Interface area") {
    Interfaces<double> interfaces = generate_interfaces();
    auto areas = interfaces.area();
    auto areas_mirror = areas.host_mirror();
    areas_mirror.deep_copy(areas);
    for (int i = 0; i < interfaces.size(); i++) {
        CHECK(Kokkos::abs(areas_mirror(i) - 1.0) < 1e-14);
    }
}

TEST_CASE("Interface directions") {
    Interfaces<double> interfaces = generate_interfaces();
    auto norms = interfaces.norm();
    auto norm_mirror = interfaces.norm().host_mirror();
    norm_mirror.deep_copy(norms);
    CHECK(Kokkos::abs(norm_mirror.x(0) - +0.0) < 1e-14);
    CHECK(Kokkos::abs(norm_mirror.y(0) - -1.0) < 1e-14);
    CHECK(Kokkos::abs(norm_mirror.z(0) - +0.0) < 1e-14);

    CHECK(Kokkos::abs(norm_mirror.x(1) - +1.0) < 1e-14);
    CHECK(Kokkos::abs(norm_mirror.y(1) - +0.0) < 1e-14);
    CHECK(Kokkos::abs(norm_mirror.z(1) - +0.0) < 1e-14);

    CHECK(Kokkos::abs(norm_mirror.x(2) - +0.0) < 1e-14);
    CHECK(Kokkos::abs(norm_mirror.y(2) - +1.0) < 1e-14);
    CHECK(Kokkos::abs(norm_mirror.z(2) - +0.0) < 1e-14);

    CHECK(Kokkos::abs(norm_mirror.x(3) - -1.0) < 1e-14);
    CHECK(Kokkos::abs(norm_mirror.y(3) - +0.0) < 1e-14);
    CHECK(Kokkos::abs(norm_mirror.z(3) - +0.0) < 1e-14);

    CHECK(Kokkos::abs(norm_mirror.x(4) - +0.0) < 1e-14);
    CHECK(Kokkos::abs(norm_mirror.y(4) - -1.0) < 1e-14);
    CHECK(Kokkos::abs(norm_mirror.z(4) - +0.0) < 1e-14);

    CHECK(Kokkos::abs(norm_mirror.x(5) - +1.0) < 1e-14);
    CHECK(Kokkos::abs(norm_mirror.y(5) - +0.0) < 1e-14);
    CHECK(Kokkos::abs(norm_mirror.z(5) - +0.0) < 1e-14);
}

TEST_CASE("Interface centres") {
    Interfaces<double> interfaces = generate_interfaces();
    std::vector<double> xs = {0.5, 1.0, 0.5, 0.0, 1.5, 2.0, 1.5, 2.5,
                              3.0, 2.5, 1.0, 0.5, 0.0, 2.0, 1.5, 3.0,
                              2.5, 1.0, 0.5, 0.0, 2.0, 1.5, 3.0, 2.5};
    std::vector<double> ys = {0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0, 0.0,
                              0.5, 1.0, 1.5, 2.0, 1.5, 1.5, 2.0, 1.5,
                              2.0, 2.5, 3.0, 2.5, 2.5, 3.0, 2.5, 3.0};
    auto centres = interfaces.centre();
    auto centre_mirror = interfaces.centre().host_mirror();
    centre_mirror.deep_copy(centres);
    for (unsigned int i = 0; i < xs.size(); i++) {
        CHECK(Kokkos::abs(centre_mirror.x(i) - xs[i]) < 1e-14);
        CHECK(Kokkos::abs(centre_mirror.y(i) - ys[i]) < 1e-14);
        CHECK(Kokkos::abs(centre_mirror.z(i) - 0.0) < 1e-14);
    }
}
