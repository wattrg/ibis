#include <parallel/parallel.h>
#include <doctest/doctest.h>
#include <doctest/extensions/doctest_mpi.h>
#include <grid/grid.h>

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

MPI_TEST_CASE("block communication", 2) {
    json config = build_config(test_rank);
    
    GridBlock<Mpi, Ibis::real> block_dev("../../../src/grid/test", config);
    auto block_host = block_dev.host_mirror();
   
}
#endif // DOCTEST_CONFIG_DISABLE

#endif // Ibis_ENABLE_MPI
