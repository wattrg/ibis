
#include <doctest/doctest.h>
#include <grid/cell.h>
#include <grid/interface.h>

struct CellInfo {
    Vertices<double> vertices;
    Interfaces<double> interfaces;
    Cells<double> cells;
};

CellInfo generate_cells() {
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
        vertices_host.positions().x(i) = vertex_pos[i].x;
        vertices_host.positions().y(i) = vertex_pos[i].y;
        vertices_host.positions().z(i) = vertex_pos[i].z;
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
    Interfaces<double> interfaces(interface_id_list, shapes);

    std::vector<std::vector<int>> cell_interfaces_list{
        {0, 1, 2, 3},     {4, 5, 6, 1},     {7, 8, 9, 5},
        {2, 10, 11, 12},  {6, 13, 14, 10},  {9, 15, 16, 13},
        {11, 17, 18, 19}, {14, 20, 21, 17}, {16, 22, 23, 20}};

    std::vector<std::vector<int>> cell_vertex_ids_raw{
        {0, 1, 5, 4},   {1, 2, 6, 5},    {2, 3, 7, 6},
        {4, 5, 9, 8},   {5, 6, 10, 9},   {6, 7, 11, 10},
        {8, 9, 13, 12}, {9, 10, 14, 13}, {10, 11, 15, 14}};

    std::vector<ElemType> cell_shapes{
        ElemType::Quad, ElemType::Quad, ElemType::Quad,
        ElemType::Quad, ElemType::Quad, ElemType::Quad,
        ElemType::Quad, ElemType::Quad, ElemType::Quad,
    };

    Cells<double> cells(cell_vertex_ids_raw, cell_interfaces_list, cell_shapes, 9, 0);
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
    auto cells_mirror = cells.host_mirror();
    cells_mirror.deep_copy(cells);

    for (int i = 0; i < cells.num_valid_cells(); i++) {
        CHECK(Kokkos::fabs(cells_mirror.volume(i) - 1.0) < 1e-14);
    }
}

TEST_CASE("cell_centre") {
    CellInfo info = generate_cells();
    Cells<double> cells = info.cells;
    Vertices<double> vertices = info.vertices;

    cells.compute_centroids(vertices);
    auto cells_mirror = cells.host_mirror();
    cells_mirror.deep_copy(cells);

    std::vector<double> x_values = {0.5, 1.5, 2.5, 0.5, 1.5,
                                    2.5, 0.5, 1.5, 2.5};
    std::vector<double> y_values = {0.5, 0.5, 0.5, 1.5, 1.5,
                                    1.5, 2.5, 2.5, 2.5};

    for (int i = 0; i < cells.num_valid_cells(); i++) {
        CHECK(Kokkos::fabs(cells_mirror.centroids().x(i) - x_values[i]) <
              1e-14);
        CHECK(Kokkos::fabs(cells_mirror.centroids().y(i) - y_values[i]) <
              1e-14);
    }
}
