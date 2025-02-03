#include <grid/grid_io.h>
#include <grid/partition.h>
#include <ibis/commands/partition/partition_grid.h>
#include <spdlog/spdlog.h>

#include <filesystem>

int partition_grid(std::string grid_filename, std::string output_dir,
                   size_t num_partitions, int argc, char** argv) {
    (void)argc;
    (void)argv;

    GridIO monolithic_grid(grid_filename);

    std::vector<GridIO> partitioned_grids;

#ifdef Ibis_ENABLE_METIS
    partitioned_grids = partition_metis(monolithic_grid, num_partitions);
#else
    spdlog::error("No partitioner is enabled");
    return 1;
#endif

    std::filesystem::path out_dir(output_dir);
    std::filesystem::create_directory(out_dir);

    for (size_t part_i = 0; part_i < partitioned_grids.size(); part_i++) {
        GridIO& partition = partitioned_grids[part_i];
        std::string part_filename("block_" + std::to_string(part_i) + ".su2");
        std::ofstream partition_file(output_dir / std::filesystem::path(part_filename));
        partition.write_su2_grid(partition_file);

        std::string cell_map_filename("cell_map_block_" + std::to_string(part_i));
        std::ofstream mapped_cells_file(output_dir /
                                        std::filesystem::path(cell_map_filename));
        partition.write_mapped_cells(mapped_cells_file);
    }

    return 0;
}
