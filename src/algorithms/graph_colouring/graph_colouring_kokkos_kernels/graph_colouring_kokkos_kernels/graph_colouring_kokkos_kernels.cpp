#include <doctest/doctest.h>
#include <graph_colouring_kokkos_kernels/graph_colouring_kokkos_kernels.h>
#include <grid/grid.h>

#include <nlohmann/json.hpp>

#include "Kokkos_UnorderedMap.hpp"
#include "util/types.h"

json build_config() {
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
    config["id"] = 0;
    config["grid_file_name"] = "grid.su2";
    return config;
}

TEST_CASE("grid colouring") {
    json config = build_config();
    using GridBlock_type = GridBlock<SharedMem, Ibis::real, Kokkos::DefaultExecutionSpace,
                                     Kokkos::DefaultExecutionSpace::array_layout>;
    GridBlock_type block(
        "../../../../../src/algorithms/graph_colouring/graph_colouring_kokkos_kernels/"
        "test",
        config);
    block.compute_graph(1);
    auto block_host = block.host_mirror();
    block_host.deep_copy(block);

    KokkosKernels_GridColourer<GridBlock_type> colourer;
    colourer.compute_colouring(block);
    Ibis::Array1D<int> colours = colourer.colours();

    auto colours_host = Kokkos::create_mirror(colours);
    Kokkos::deep_copy(colours_host, colours);

    for (size_t cell_i = 0; cell_i < block_host.num_cells(); cell_i++) {
        auto neighbour_cells = block_host.cells().neighbour_cells(cell_i);
        for (size_t neighbour_i = 0; neighbour_i < neighbour_cells.size();
             neighbour_i++) {
            size_t neighbour_cell = neighbour_cells(neighbour_i);
            if (neighbour_cell >= block_host.num_cells()) {
                continue;
            }
            CHECK(colours_host(cell_i) != colours_host(neighbour_cell));

            // Check that the neighbour's neighbours also don't have the same colour
            auto neighbour_neighbours =
                block_host.cells().neighbour_cells(neighbour_cell);
            size_t num_neighbour_neighbours = neighbour_neighbours.size();
            for (size_t neighbour_neighbour_i = 0;
                 neighbour_neighbour_i < num_neighbour_neighbours;
                 neighbour_neighbour_i++) {
                size_t neighbour_neighbour_cell =
                    neighbour_neighbours(neighbour_neighbour_i);
                if (neighbour_neighbour_cell != cell_i &&
                    neighbour_neighbour_cell < block_host.num_cells()) {
                    CHECK(colours_host(cell_i) != colours_host(neighbour_neighbour_cell));
                }
            }
        }
    }
}
