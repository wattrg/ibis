#ifndef CRS_H
#define CRS_H

#include <util/types.h>

namespace Ibis {

template <typename OffsetType, typename OrdinalType, class Layout = DefaultArrayLayout,
          class MemSpace = DefaultMemSpace>
class CrsGraph {
public:
    using layout = Layout;
    using offset = OffsetType;
    using ordinal = OrdinalType;
    using mem_space = MemSpace;
    
    CrsGraph() {}

    CrsGraph(Array1D<OffsetType, Layout, MemSpace> row_map_,
             Array1D<OrdinalType, Layout, MemSpace> entries_)
        : row_map(row_map_), entries(entries_) {}

    OrdinalType entry_index(OffsetType row, OffsetType col) const {
        OffsetType row_start_idx = row_map(row);
        OffsetType num_entries_in_row = row_map(row + 1) - row_start_idx;
        for (OffsetType col_idx = 0; col_idx < num_entries_in_row; col_idx++) {
            if (entries(row_start_idx + col_idx) == col) {
                return row_start_idx + col_idx;
            }
        }
    }

    size_t num_entries() const { return entries.size(); }

    size_t num_rows() const { return row_map.size() - 1; }    

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

    CrsMatrix() {}

    CrsMatrix(CrsGraph<OffsetType, OrdinalType, Layout, MemSpace> graph_,
              Array1D<ScalarType, Layout, MemSpace> values_) {
        graph = graph_;
        values = values_;
    }

    KOKKOS_INLINE_FUNCTION
    ScalarType& operator()(OffsetType row, OffsetType col) {
        return values(graph.entry_index(row, col));
    }

    KOKKOS_INLINE_FUNCTION
    ScalarType& operator()(OffsetType row, OffsetType col) const {
        return values(graph.entry_index(row, col));
    }

    size_t num_entries() const { return graph.num_entries(); }

    size_t num_rows() const { return graph.num_rows(); }

public:
    CrsGraph<OffsetType, OrdinalType, Layout, MemSpace> graph;
    Array1D<ScalarType, Layout, MemSpace> values;
};

}  // namespace Ibis

#endif
