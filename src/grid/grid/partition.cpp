#include <metis.h>
#include <grid/partition.h>

#ifdef Ibis_ENABLE_METIS
std::vector<GridIO> partition_metis(GridIO& monolithic_grid, size_t n_partitions) {
    // construct information in the form METIS expects it
    idx_t ne = monolithic_grid.cells().size();
    idx_t nn = monolithic_grid.vertices().size();
    idx_t ncommon = monolithic_grid.dim();
    idx_t nparts = n_partitions;

    std::vector<idx_t> eptr{0}; // size ne + 1
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
    int metis_result = METIS_MeshToDual(&ne, &nn, eptr.data(), eind.data(),
                                        &ncommon, &numflag, &xadj, &adjncy);

    // partition the graph
    idx_t ncon = 1;
    idx_t objval;
    std::vector<idx_t> partitions(ne);
    metis_result = METIS_PartGraphKway(&ne, &ncon, xadj, adjncy, NULL, NULL, NULL,
                                       &nparts, NULL, NULL, NULL, &objval,
                                       partitions.data());
    
    // build the partitioned grids
    std::vector<GridIO> grids(nparts);
    std::vector<std::vector<size_t>> cells_in_partition(n_partitions);
    for (size_t cell_i = 0; cell_i < monolithic_grid.cells().size(); cell_i++) {
        size_t cell_i_partition = partitions[cell_i];
        cells_in_partition[cell_i_partition].push_back(cell_i);
    }
    for (size_t partition_i = 0; partition_i < n_partitions; partition_i++) {
        grids[partition_i] = GridIO(monolithic_grid, cells_in_partition[partition_i]);
    }
    

    // METIS allocated memory for xadj and adjncy, but we are responsible
    // for cleaning up the memory.
    // I don't know what the difference between METIS_Free and plain free is...
    METIS_Free(xadj);
    METIS_Free(adjncy);
    return grids;
}
#endif
