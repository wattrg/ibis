#include <doctest/doctest.h>
#include <doctest/extensions/doctest_mpi.h>
#include <grid/grid.h>
#include <parallel/parallel.h>

#ifdef Ibis_ENABLE_MPI
#ifndef DOCTEST_CONFIG_DISABLE

json build_config(size_t id) {
    json config{};
    json boundaries{};
    json slip_wall{};
    json inflow{};
    json outflow{};
    json motion{};
    slip_wall["ghost_cells"] = true;
    inflow["ghost_cells"] = true;
    outflow["ghost_cells"] = true;
    boundaries["slip_wall_bottom"] = slip_wall;
    boundaries["slip_wall_top"] = slip_wall;
    boundaries["inflow"] = inflow;
    boundaries["outflow"] = outflow;
    config["boundaries"] = boundaries;
    motion["enabled"] = false;
    config["motion"] = motion;
    config["id"] = id;
    config["grid_file_name"] = std::format("grid_{:04}.su2", id);
    config["cell_map_file_name"] = std::format("mapped_cells_{:04}", id);
    return config;
}

MPI_TEST_CASE("block volume communication", 2) {
    json config = build_config(test_rank);

    GridBlock<Mpi, Ibis::real> block_dev("../../../src/grid/test", config);
    auto block_host = block_dev.host_mirror();
    block_host.deep_copy(block_dev);

    CHECK(block_dev.other_blocks().size() == 1);
    CHECK(block_host.other_blocks().size() == 1);

    for (size_t cell_i = 0; cell_i < block_host.num_total_cells(); cell_i++) {
        CHECK(block_host.cells().volume(cell_i) == doctest::Approx(1.0));
    }
}

MPI_TEST_CASE("block position communication", 2) {
    json config = build_config(test_rank);

    GridBlock<Mpi, Ibis::real> block_dev("../../../src/grid/test", config);
    auto block_host = block_dev.host_mirror();
    block_host.deep_copy(block_dev);

    size_t other_block = test_rank ^= 1;
    json other_config = build_config(other_block);
    GridBlock<Mpi, Ibis::real> block_other_dev("../../../src/grid/test", config);
    auto block_other_host = block_other_dev.host_mirror();
    block_other_host.deep_copy(block_other_dev);

    for (size_t cell_i = 0;
         cell_i < block_host.internal_boundary_cells(other_block).size(); cell_i++) {
        size_t other_cell_id =
            block_host.internal_boundary_external_cells(other_block)(cell_i);
        size_t local_cell_id = block_host.internal_boundary_cells(other_block)(cell_i);
        CHECK(block_host.cells().centroids().x(local_cell_id) ==
              doctest::Approx(block_other_host.cells().centroids().x(other_cell_id)));
        CHECK(block_host.cells().centroids().y(local_cell_id) ==
              doctest::Approx(block_other_host.cells().centroids().y(other_cell_id)));
        CHECK(block_host.cells().centroids().z(local_cell_id) ==
              doctest::Approx(block_other_host.cells().centroids().z(other_cell_id)));
    }
}
#endif  // DOCTEST_CONFIG_DISABLE

#endif  // Ibis_ENABLE_MPI
