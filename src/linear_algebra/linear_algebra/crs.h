#ifndef CRS_H
#define CRS_H

#include <util/types.h>

namespace Ibis {

template <typename OffsetType, typename OrdinalType, typename ScalarType,
          class Layout = DefaultArrayLayout, class MemSpace = DefaultMemSpace>
class CrsMatrix {
public:
    CrsMatrix();

    CrsMatrix(Ibis::Array1D<OffsetType, Layout, MemSpace> row_map,
              Ibis::Array1D<OrdinalType, Layout, MemSpace> entries,
              Ibis::Array1D<ScalarType, Layout, MemSpace> values);

    Ibis::Array1D<OffsetType, Layout, MemSpace> row_map() { return row_map_; }
    Ibis::Array1D<OrdinalType, Layout, MemSpace> entries() { return entries_; }
    Ibis::Array1D<ScalarType, Layout, MemSpace> values() { return values_; }

    KOKKOS_INLINE_FUNCTION
    ScalarType& operator()(OffsetType row, OffsetType col) {
        OffsetType row_start_idx = row_map_(row);
        OffsetType num_entries_in_row = row_map_(row + 1) - row_start_idx;
        OrdinalType value_idx;
        for (OffsetType col_idx = 0; col_idx < num_entries_in_row; col_idx++) {
            if (entries_(row_start_idx + col_idx) == col) {
                return values_(row_start_idx + col_idx);
            }
        }
    }

    KOKKOS_INLINE_FUNCTION
    ScalarType& operator()(OffsetType row, OffsetType col) const {
        OffsetType row_start_idx = row_map_(row);
        OffsetType num_entries_in_row = row_map_(row + 1) - row_start_idx;
        OrdinalType value_idx;
        for (OffsetType col_idx = 0; col_idx < num_entries_in_row; col_idx++) {
            if (entries_(row_start_idx + col_idx) == col) {
                return values_(row_start_idx + col_idx);
            }
        }
    }

    size_t num_entries() const { return entries_.size(); }

    size_t num_rows() const { return row_map_.size() - 1; }

private:
    Ibis::Array1D<OffsetType, Layout, MemSpace> row_map_;
    Ibis::Array1D<OrdinalType, Layout, MemSpace> entries_;
    Ibis::Array1D<ScalarType, Layout, MemSpace> values_;
};

}  // namespace Ibis

#endif
