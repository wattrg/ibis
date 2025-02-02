#ifndef PARTITION_GRID_H
#define PARTITION_GRID_H

#include <CLI/CLI.hpp>

int partition_grid(std::string grid_filename, std::string output_dir,
                   size_t num_partitions, int argc, char* argv[]);
#endif
