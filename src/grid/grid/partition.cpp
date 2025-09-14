#ifdef Ibis_ENABLE_METIS

#include <doctest/doctest.h>
#include <grid/partition.h>
#include <metis.h>
#include <spdlog/spdlog.h>

void check_metis_result(int result) {
    if (result == METIS_ERROR_INPUT) {
        spdlog::error("Metis input error");
        throw new std::runtime_error("Metis input error");
    } else if (result == METIS_ERROR_MEMORY) {
        spdlog::error("Metis ran out of memory");
        throw new std::runtime_error("Metis ran out of memory");
    } else if (result == METIS_ERROR) {
        spdlog::error("Metis error");
        throw new std::runtime_error("Metis error");
    }
}

std::vector<GridIO> partition_metis(GridIO& monolithic_grid, size_t n_partitions) {
    // construct information in the form METIS expects it
    idx_t ne = monolithic_grid.cells().size();
    idx_t nn = monolithic_grid.vertices().size();
    idx_t ncommon = monolithic_grid.dim();
    idx_t nparts = n_partitions;

    std::vector<idx_t> eptr{0};  // size ne + 1
    std::vector<idx_t> eind;
    std::vector<ElemIO> cells = monolithic_grid.cells();
    for (idx_t i = 0; i < ne; i++) {
        ElemIO cell = cells[i];
        std::vector<size_t> vertex_ids = cell.vertex_ids();
        eptr.push_back(eptr.back() + vertex_ids.size());
        for (size_t vertex_id : vertex_ids) {
            eind.push_back(vertex_id);
        }
    }

    // convert the mesh to a dual graph, so we have the connectivity information
    // for building the separate grids later on
    idx_t numflag = 0;
    idx_t* xadj;
    idx_t* adjncy;
    int metis_result = METIS_MeshToDual(&ne, &nn, eptr.data(), eind.data(), &ncommon,
                                        &numflag, &xadj, &adjncy);
    check_metis_result(metis_result);

    // partition the graph
    idx_t ncon = 1;
    idx_t objval;
    std::vector<idx_t> partitions(ne);
    metis_result =
        METIS_PartGraphKway(&ne, &ncon, xadj, adjncy, NULL, NULL, NULL, &nparts, NULL,
                            NULL, NULL, &objval, partitions.data());
    check_metis_result(metis_result);

    // build lists of cells in each partition
    std::vector<std::vector<size_t>> cells_in_partition(n_partitions);
    std::vector<std::unordered_map<size_t, size_t>> global_local_cell_map(n_partitions);
    for (size_t cell_i = 0; cell_i < monolithic_grid.cells().size(); cell_i++) {
        size_t cell_i_partition = partitions[cell_i];
        cells_in_partition[cell_i_partition].push_back(cell_i);
        global_local_cell_map[cell_i_partition][cell_i] =
            cells_in_partition[cell_i_partition].size() - 1;
    }

    // build cell mappings
    // The approach is to loop through all the cells, and check if its neighbours are in
    // the same partition as it. If they are not, build the cell mapping.
    // Later on, the neighbour will be visted, and the symmetric mapping will be built.
    std::vector<std::vector<CellMapping>> cell_mapping(n_partitions);
    for (size_t cell_i = 0; cell_i < monolithic_grid.cells().size(); cell_i++) {
        size_t this_cell_partition = partitions[cell_i];
        size_t ngbr_idx_start = xadj[cell_i];
        size_t ngbr_idx_end = xadj[cell_i + 1];
        size_t num_ngbr = ngbr_idx_end - ngbr_idx_start;
        for (size_t ngbr_i = 0; ngbr_i < num_ngbr; ngbr_i++) {
            size_t ngbr_id = adjncy[ngbr_idx_start + ngbr_i];
            size_t ngbr_partition = partitions[ngbr_id];
            if (this_cell_partition != ngbr_partition) {
                // The ids of the two cells in their local block
                size_t local_cell_id = global_local_cell_map[this_cell_partition][cell_i];
                size_t other_cell_id = global_local_cell_map[ngbr_partition][ngbr_id];

                // find the common interface between the two cells
                // const ElemIO& local_cell = monolithic_grid.cells()[cell_i];
                // const ElemIO& other_cell = monolithic_grid.cells()[ngbr_id];
                // std::vector<ElemIO> other_faces = other_cell.interfaces();
                std::vector<ElemIO> local_faces = monolithic_grid.cells()[cell_i].interfaces();
                std::vector<ElemIO> other_faces = monolithic_grid.cells()[ngbr_id].interfaces();

                const InterfaceLookup& face_lookup = monolithic_grid.interface_lookup();
                size_t common_face_id = std::numeric_limits<size_t>::max();
                for (const ElemIO& local_face : local_faces) {
                    size_t local_face_id = face_lookup.id(local_face.vertex_ids());
                    for (const ElemIO& other_face : other_faces) {
                        size_t other_face_id = face_lookup.id(other_face.vertex_ids());
                        if (local_face_id == other_face_id) {
                            // we've found the face!
                            common_face_id = local_face_id;
                        }
                    }
                }
                if (common_face_id == std::numeric_limits<size_t>::max()) {
                    spdlog::error("Unable to find common interface for Mapped cells");
                    throw std::runtime_error(
                        "Unable to find common interface for Mapped cells");
                }

                cell_mapping[this_cell_partition].push_back(CellMapping(
                    local_cell_id, ngbr_partition, other_cell_id, common_face_id));
            }
        }
    }

    // METIS allocated memory for xadj and adjncy, but we are responsible
    // for cleaning up the memory. We don't need them anymore, so we'll
    // free the memory now.
    // I don't know what the difference between METIS_Free and plain free is...
    METIS_Free(xadj);
    METIS_Free(adjncy);

    // Finally, build the GridIO objects
    std::vector<GridIO> grids(n_partitions);
    for (size_t partition_i = 0; partition_i < n_partitions; partition_i++) {
        grids[partition_i] = GridIO(monolithic_grid, cells_in_partition[partition_i],
                                    std::move(cell_mapping[partition_i]), partition_i);
    }

    return grids;
}

#ifndef DOCTEST_CONFIG_DISABLE
std::vector<GridIO> build_partitioned_grid() {
    // read a test grid
    GridIO monolithic_grid("../../../src/grid/test/grid.su2");

    // partition it into two parts
    return partition_metis(monolithic_grid, 2);
}

TEST_CASE("partition_metis_number_of_cells") {
    std::vector<GridIO> partitioned_grids = build_partitioned_grid();

    size_t total_cells = 0;
    for (const GridIO& partition : partitioned_grids) {
        total_cells += partition.cells().size();
    }

    CHECK(total_cells == 9);
}

TEST_CASE("partition_metis_symmetric_mapping") {
    std::vector<GridIO> partitioned_grids = build_partitioned_grid();

    std::vector<std::vector<CellMapping>> mappings;
    for (const GridIO& partition : partitioned_grids) {
        mappings.push_back(partition.cell_mapping());
    }

    for (size_t this_block = 0; this_block < mappings.size(); this_block++) {
        const std::vector<CellMapping>& part_mapping = mappings[this_block];
        for (const CellMapping& map : part_mapping) {
            size_t this_cell = map.local_cell;
            size_t other_block = map.other_block;
            size_t other_cell = map.other_cell;

            bool mapping_is_symmetric = false;
            for (const CellMapping& other_map : mappings[other_block]) {
                if (other_map.other_block == this_block &&
                    other_map.local_cell == other_cell &&
                    other_map.other_cell == this_cell) {
                    mapping_is_symmetric = true;
                }
            }
            CHECK(mapping_is_symmetric);
        }
    }
}
#endif  // DOCTEST_CONFIG_DISABLE
#endif  // Ibis_ENABLE_METIS
