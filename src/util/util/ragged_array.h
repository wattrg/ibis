#ifndef RAGGED_ARRAY_H
#define RAGGED_ARRAY_H

#include <string>
#include <vector>
#include <Kokkos_Core.hpp>

namespace Ibis {

template<typename DataType, 
         class Layout=Kokkos::DefaultExecutionSpace::array_layout,
         class Space=Kokkos::DefaultExecutionSpace::memory_space>
class RaggedArray{
public:
    using ArrayType = Kokkos::View<DataType*, Layout, Space>;
    using host_layout = Layout;
    using host_space = Kokkos::DefaultHostExecutionSpace;
    using HostMirror = RaggedArray<DataType, host_layout, host_space>;

private:
    using OffsetType = Kokkos::View<int*, Layout, Space>;

    // allow accessing the private data of the same class with differe
    // template parameters (e.g. living in different memory spaces)
    template <typename OtherDataType, class OtherLayout, class OtherSpace>
    friend class RaggedArray;

public:
    RaggedArray() {}

    RaggedArray(ArrayType data, OffsetType offsets) 
        : data_(data), offsets_(offsets)
    {}

    RaggedArray(std::string label, std::vector<std::vector<DataType>> data) {
        // count the total number of entries
        unsigned int n = 0;
        for (const auto &row: data) {
            n += row.size(); 
        }

        // allocate memory
        data_ = ArrayType(label, n);
        offsets_ = OffsetType("offsets", data.size()+1);

        // initialise memory (on the CPU)
        auto data_host = Kokkos::create_mirror_view(data_);
        auto offsets_host = Kokkos::create_mirror_view(offsets_);
        int i = 0;
        int row_i = 0;
        for (const auto &row: data) {
            offsets_host(row_i) = i;
            for (const auto &value: row) {
                data_host(i) = value;
                i++;
            }
            row_i++;
        }
        offsets_host(row_i) = i;

        // Copy data from CPU to the GPU (if needed)
        Kokkos::deep_copy(data_, data_host);
        Kokkos::deep_copy(offsets_, offsets_host);
    }

    // Read-only value at (row, col)
    KOKKOS_INLINE_FUNCTION
    DataType & operator() (const int row, const int col) {
        int index = offsets_(row) + col;  
        return data_(index);
    } 

    // Read-write value at (row, col)
    KOKKOS_INLINE_FUNCTION
    const DataType & operator() (const int row, const int col) const {
        int index = offsets_(row) + col;  
        return data_(index);
    } 

    // Read-write view to a particular row
    KOKKOS_INLINE_FUNCTION
    auto operator() (const int row) {
        int first = offsets_(row);
        int last = offsets_(row+1);
        return Kokkos::subview(data_, Kokkos::make_pair(first, last));
    }

    // Read-only view to a particular row
    KOKKOS_INLINE_FUNCTION
    auto operator() (const int row) const {
        int first = offsets_(row);
        int last = offsets_(row+1);
        return Kokkos::subview(data_, Kokkos::make_pair(first, last));
    }

    // Create a host mirror
    HostMirror host_mirror() {
        auto data_mirror = Kokkos::create_mirror_view(data_);
        auto offsets_mirror = Kokkos::create_mirror_view(offsets_);
        return HostMirror(data_mirror, offsets_mirror); 
    }

    // Create a host mirror and copy contents to the mirror
    HostMirror host_mirror_and_copy() {
        auto data_mirror = Kokkos::create_mirror_view(data_);
        auto offsets_mirror = Kokkos::create_mirror_view(offsets_);
        HostMirror host_mirror (data_mirror, offsets_mirror);
        host_mirror.deep_copy(*this);
        return host_mirror;
    }

    // Copy data from another ragged array (with the same dimensions)
    template <class OtherSpace>
    void deep_copy(const RaggedArray<DataType, Layout, OtherSpace> &other) {
        Kokkos::deep_copy(data_, other.data_); 
        Kokkos::deep_copy(offsets_, other.offsets_);
    }

private:
    ArrayType data_;
    OffsetType offsets_;
};

}

#endif
