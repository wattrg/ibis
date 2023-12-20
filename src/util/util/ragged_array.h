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

public:
    RaggedArray() {}

    RaggedArray(std::string label, const int n) {
        data_ = ArrayType(label, n);
    }

    RaggedArray(std::string label, 
                std::vector<std::vector<DataType>> data) {
        // count the total number of entries
        unsigned int n = 0;
        for (const auto &row: data) {
            n += row.size(); 
        }

        // allocate memory
        data_ = ArrayType(label, n);
        offsets_ = Kokkos::View<int*, Layout, Space>("offsets", data.size()+1);

        // initialise memory
        auto data_host = Kokkos::create_mirror(data_);
        auto offsets_host = Kokkos::create_mirror(offsets_);

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

        Kokkos::deep_copy(data_, data_host);
        Kokkos::deep_copy(offsets_, offsets_host);
    }

    DataType operator() (const int row, const int col) {
        int index = offsets_(row) + col;  
        return data_(index);
    } 

private:
    ArrayType data_;
    Kokkos::View<int*, Layout, Space> offsets_;
};

}

#endif
