#ifndef GRID_PARTITION_H
#define GRID_PARTITION_H

#include <grid/grid_io.h>

std::vector<GridIO> partition_metis(GridIO& monolithic_grid, size_t n_partitions);

#endif
