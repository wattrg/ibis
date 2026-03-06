#ifndef CRS_H
#define CRS_H

#include <util/types.h>

#include <iostream>

namespace Ibis {

template <typename OffsetType, typename OrdinalType, class Layout = DefaultArrayLayout,
          class MemSpace = DefaultMemSpace>
class CrsGraph {
public:
    using layout = Layout;
    using offset = OffsetType;
    using ordinal = OrdinalType;
    using mem_space = MemSpace;
    using rowmap_type = Array1D<OffsetType, Layout, MemSpace>;
    using entries_type = Array1D<OrdinalType, Layout, MemSpace>;
    using host_mirror_type =
        CrsGraph<OffsetType, OrdinalType, Layout, Ibis::DefaultHostMemSpace>;

    CrsGraph() {}

    CrsGraph(Array1D<OffsetType, Layout, MemSpace> row_map_,
             Array1D<OrdinalType, Layout, MemSpace> entries_)
        : row_map(row_map_), entries(entries_) {}

    CrsGraph(OffsetType num_rows, OrdinalType num_entries) {
        row_map =
            Array1D<OffsetType, Layout, MemSpace>("CrsGraph::row_map", num_rows + 1);
        entries =
            Array1D<OrdinalType, Layout, MemSpace>("CrsGraph::entries", num_entries);
    }

    KOKKOS_INLINE_FUNCTION
    OrdinalType entry_index(OffsetType row, OffsetType col) const {
        OffsetType row_start_idx = row_map(row);
        OffsetType num_entries_in_row = row_map(row + 1) - row_start_idx;
        for (OffsetType col_idx = 0; col_idx < num_entries_in_row; col_idx++) {
            if (entries(row_start_idx + col_idx) == col) {
                return row_start_idx + col_idx;
            }
        }
        assert(false && "row and col is not present in CrsMatrix");
    }

    KOKKOS_INLINE_FUNCTION
    size_t num_entries() const { return entries.size(); }

    KOKKOS_INLINE_FUNCTION
    size_t num_rows() const { return row_map.size() - 1; }

    host_mirror_type host_mirror() { return host_mirror_type(num_rows(), num_entries()); }

    template <class OtherMemSpace>
    void deep_copy(
        const CrsGraph<OffsetType, OrdinalType, Layout, OtherMemSpace>& other) {
        Kokkos::deep_copy(row_map, other.row_map);
        Kokkos::deep_copy(entries, other.entries);
    }

public:
    Ibis::Array1D<OffsetType, Layout, MemSpace> row_map;
    Ibis::Array1D<OrdinalType, Layout, MemSpace> entries;
};

template <typename OffsetType, typename OrdinalType, typename ScalarType,
          class Layout = DefaultArrayLayout, class MemSpace = DefaultMemSpace>
class CrsMatrix {
public:
    using layout = Layout;
    using offset = OffsetType;
    using ordinal = OrdinalType;
    using scalar = ScalarType;
    using mem_space = MemSpace;
    using host_mirror_type =
        CrsMatrix<OffsetType, OrdinalType, ScalarType, Layout, Ibis::DefaultHostMemSpace>;

    CrsMatrix() {}

    CrsMatrix(CrsGraph<OffsetType, OrdinalType, Layout, MemSpace> graph_,
              Array1D<ScalarType, Layout, MemSpace> values_) {
        graph = graph_;
        values = values_;
    }

    CrsMatrix(OffsetType num_rows, OrdinalType num_entries) {
        graph =
            CrsGraph<OffsetType, OrdinalType, Layout, MemSpace>(num_rows, num_entries);
        values = Array1D<ScalarType, Layout, MemSpace>("CrsMatrix::values", num_entries);
    }

    KOKKOS_INLINE_FUNCTION
    ScalarType& operator()(OffsetType row, OffsetType col) {
        return values(graph.entry_index(row, col));
    }

    KOKKOS_INLINE_FUNCTION
    ScalarType& operator()(OffsetType row, OffsetType col) const {
        return values(graph.entry_index(row, col));
    }

    KOKKOS_INLINE_FUNCTION
    size_t num_entries() const { return graph.num_entries(); }

    KOKKOS_INLINE_FUNCTION
    size_t num_rows() const { return graph.num_rows(); }

    host_mirror_type host_mirror() { return host_mirror_type(num_rows(), num_entries()); }

    template <class OtherMemSpace>
    void deep_copy(const CrsMatrix<OffsetType, OrdinalType, ScalarType, Layout,
                                   OtherMemSpace>& other) {
        graph.deep_copy(other.graph);
        Kokkos::deep_copy(values, other.values);
    }

public:
    CrsGraph<OffsetType, OrdinalType, Layout, MemSpace> graph;
    Array1D<ScalarType, Layout, MemSpace> values;
};

}  // namespace Ibis

#endif
