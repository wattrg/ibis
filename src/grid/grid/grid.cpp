
#include <doctest/doctest.h>
#include <grid/grid.h>
#include <grid/grid_io.h>
#include <grid/interface.h>

struct GridInfo {
    Vertices<double, Kokkos::DefaultHostExecutionSpace> vertices;
    Interfaces<double, Kokkos::DefaultHostExecutionSpace> faces;
    Cells<double, Kokkos::DefaultHostExecutionSpace> cells;
};

GridInfo build_test_grid() {
    GridInfo grid_info{};
    Vertices<double, Kokkos::DefaultHostExecutionSpace> vertices(16);
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
        vertices.set_vertex_position(i, vertex_pos[i]);
    }

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

    Interfaces<double, Kokkos::DefaultHostExecutionSpace> interfaces(
        interface_id_list, shapes);

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

    Cells<double, Kokkos::DefaultHostExecutionSpace> cells(
        cell_vertex_ids_raw, cell_interfaces_list, cell_shapes, 9, 0);
    grid_info.vertices = vertices;
    grid_info.faces = interfaces;
    grid_info.cells = cells;
    return grid_info;
}

json build_config() {
    json config{};
    json boundaries{};
    json slip_wall{};
    json inflow{};
    json outflow{};
    slip_wall["ghost_cells"] = true;
    inflow["ghost_cells"] = true;
    outflow["ghost_cells"] = true;
    boundaries["slip_wall_bottom"] = slip_wall;
    boundaries["slip_wall_top"] = slip_wall;
    boundaries["inflow"] = inflow;
    boundaries["outflow"] = outflow;
    config["boundaries"] = boundaries;
    return config;
}

TEST_CASE("grid vertices") {
    GridInfo expected = build_test_grid();
    json config = build_config();
    GridBlock<double> block_dev("../../../src/grid/test/grid.su2", config);
    auto block = block_dev.host_mirror();
    block.deep_copy(block_dev);
    auto expected_vertices = expected.vertices;
    CHECK(block.vertices() == expected_vertices);
}

TEST_CASE("grid interfaces") {
    GridInfo expected = build_test_grid();
    json config = build_config();
    GridBlock<double> block_dev("../../../src/grid/test/grid.su2", config);
    auto block = block_dev.host_mirror();
    block.deep_copy(block_dev);
    CHECK(block.interfaces() == expected.faces);
}

TEST_CASE("grid cell faces") {
    GridInfo expected = build_test_grid();
    json config = build_config();
    GridBlock<double> block_dev("../../../src/grid/test/grid.su2", config);
    auto block = block_dev.host_mirror();
    block.deep_copy(block_dev);
    CHECK(block.cells().num_valid_cells() == expected.cells.num_valid_cells());
    for (int i = 0; i < block.num_cells(); i++) {
        for (unsigned int j = 0; j < block.cells().faces().face_ids(i).size();
             j++) {
            CHECK(block.cells().faces().face_ids(i)(j) ==
                  expected.cells.faces().face_ids(i)(j));
        }
    }
}

TEST_CASE("grid cell faces 2") {
    json config = build_config();
    GridBlock<double> block_dev("../../../src/grid/test/grid.su2", config);
    auto block = block_dev.host_mirror();
    block.deep_copy(block_dev);
    std::vector<std::vector<int>> face_ids = {
        {0, 1, 2, 3},     {4, 5, 6, 1},     {7, 8, 9, 5},
        {2, 10, 11, 12},  {6, 13, 14, 10},  {9, 15, 16, 13},
        {11, 17, 18, 19}, {14, 20, 21, 17}, {16, 22, 23, 20}};
    CHECK(block.num_cells() == 9);
    for (int i = 0; i < block.num_cells(); i++) {
        CellFaces<double>::mirror_type faces = block.cells().faces();
        for (unsigned int j = 0; j < faces.face_ids(i).size(); j++) {
            CHECK(faces.face_ids(i)(j) == face_ids[i][j]);
        }
    }
}

TEST_CASE("grid cell outsigns") {
    GridInfo expected = build_test_grid();
    json config = build_config();
    GridBlock<double> block_dev("../../../src/grid/test/grid.su2", config);
    auto block = block_dev.host_mirror();
    block.deep_copy(block_dev);
    CHECK(block.cells().num_valid_cells() == expected.cells.num_valid_cells());
    std::vector<std::vector<int>> outsigns = {
        {1, 1, 1, 1},  {1, 1, 1, -1},  {1, 1, 1, -1},
        {-1, 1, 1, 1}, {-1, 1, 1, -1}, {-1, 1, 1, -1},
        {-1, 1, 1, 1}, {-1, 1, 1, -1}, {-1, 1, 1, -1}};
    for (int i = 0; i < block.cells().num_valid_cells(); i++) {
        for (unsigned int j = 0; j < block.cells().faces().outsigns(i).size();
             j++) {
            CHECK(block.cells().faces().outsigns(i)(j) == outsigns[i][j]);
        }
    }
}

TEST_CASE("ghost cell centres") {
    json config = build_config();
    GridBlock<double> block_dev("../../../src/grid/test/grid.su2", config);
    auto block_host = block_dev.host_mirror();
    block_host.deep_copy(block_dev);

    // test the inflow cells
    auto inflow_ghost_cells = block_host.ghost_cells("inflow");
    int ghost_cell = inflow_ghost_cells(0);
    CHECK(block_host.cells().centroids().x(ghost_cell) == -0.5);
    CHECK(block_host.cells().centroids().y(ghost_cell) == 0.5);
    CHECK(block_host.cells().centroids().z(ghost_cell) == 0.0);

    ghost_cell = inflow_ghost_cells(1);
    CHECK(block_host.cells().centroids().x(ghost_cell) == -0.5);
    CHECK(block_host.cells().centroids().y(ghost_cell) == 1.5);
    CHECK(block_host.cells().centroids().z(ghost_cell) == 0.0);

    ghost_cell = inflow_ghost_cells(2);
    CHECK(block_host.cells().centroids().x(ghost_cell) == -0.5);
    CHECK(block_host.cells().centroids().y(ghost_cell) == 2.5);
    CHECK(block_host.cells().centroids().z(ghost_cell) == 0.0);

    // test slip_wall_bottom
    auto bottom_ghost_cells = block_host.ghost_cells("slip_wall_bottom");
    ghost_cell = bottom_ghost_cells(0);
    CHECK(block_host.cells().centroids().x(ghost_cell) == 0.5);
    CHECK(block_host.cells().centroids().y(ghost_cell) == -0.5);
    CHECK(block_host.cells().centroids().z(ghost_cell) == 0.0);

    ghost_cell = bottom_ghost_cells(1);
    CHECK(block_host.cells().centroids().x(ghost_cell) == 1.5);
    CHECK(block_host.cells().centroids().y(ghost_cell) == -0.5);
    CHECK(block_host.cells().centroids().z(ghost_cell) == 0.0);

    ghost_cell = bottom_ghost_cells(2);
    CHECK(block_host.cells().centroids().x(ghost_cell) == 2.5);
    CHECK(block_host.cells().centroids().y(ghost_cell) == -0.5);
    CHECK(block_host.cells().centroids().z(ghost_cell) == 0.0);

    // test outflow
    auto outflow_ghost_cells = block_host.ghost_cells("outflow");
    ghost_cell = outflow_ghost_cells(0);
    CHECK(block_host.cells().centroids().x(ghost_cell) == 3.5);
    CHECK(block_host.cells().centroids().y(ghost_cell) == 0.5);
    CHECK(block_host.cells().centroids().z(ghost_cell) == 0.0);

    ghost_cell = outflow_ghost_cells(1);
    CHECK(block_host.cells().centroids().x(ghost_cell) == 3.5);
    CHECK(block_host.cells().centroids().y(ghost_cell) == 1.5);
    CHECK(block_host.cells().centroids().z(ghost_cell) == 0.0);

    ghost_cell = outflow_ghost_cells(2);
    CHECK(block_host.cells().centroids().x(ghost_cell) == 3.5);
    CHECK(block_host.cells().centroids().y(ghost_cell) == 2.5);
    CHECK(block_host.cells().centroids().z(ghost_cell) == 0.0);

    // test slip_wall_top
    auto top_ghost_cells = block_host.ghost_cells("slip_wall_top");
    ghost_cell = top_ghost_cells(0);
    CHECK(block_host.cells().centroids().x(ghost_cell) == 0.5);
    CHECK(block_host.cells().centroids().y(ghost_cell) == 3.5);
    CHECK(block_host.cells().centroids().z(ghost_cell) == 0.0);

    ghost_cell = top_ghost_cells(1);
    CHECK(block_host.cells().centroids().x(ghost_cell) == 1.5);
    CHECK(block_host.cells().centroids().y(ghost_cell) == 3.5);
    CHECK(block_host.cells().centroids().z(ghost_cell) == 0.0);

    ghost_cell = top_ghost_cells(2);
    CHECK(block_host.cells().centroids().x(ghost_cell) == 2.5);
    CHECK(block_host.cells().centroids().y(ghost_cell) == 3.5);
    CHECK(block_host.cells().centroids().z(ghost_cell) == 0.0);
}
