#ifndef KOKKOS_KERNEL_GRAPH_COLOUR_H
#define KOKKOS_KERNEL_GRAPH_COLOUR_H

#include <graph_colouring/graph_colouring.h>
// #include <grid/grid.h>

#include <KokkosGraph_Distance2Color.hpp>

using Scalar = KokkosKernels::default_scalar;
using Ordinal = KokkosKernels::default_lno_t;
using Offset = KokkosKernels::default_size_type;

template <class GridBlock_type>
class GridColourer;

template <class GridBlock_type>
class KokkosKernels_GridColourer : GridColourer<GridBlock_type> {
public:
    Ibis::Array1D<Ordinal> colours() const {
        return colours_;
    }

    size_t num_colours() const { return num_colours_; }

    void compute_colouring(const GridBlock_type& grid) {
        // using Device = ExecSpace::device_type;
        using MemSpace = GridBlock_type::memory_space;
        using ExecSpace = GridBlock_type::execution_space;

        // TODO: Think about constructing the CRS matrix on the GPU
        auto grid_host = grid.host_mirror();
        grid_host.deep_copy(grid);
        std::vector<Offset> serial_row_map{0};
        std::vector<Ordinal> serial_entries;
        for (size_t cell_i = 0; cell_i < grid_host.num_cells(); cell_i++) {
            auto neighbour_cells = grid_host.cells().neighbour_cells(cell_i);
            for (size_t neighbour_i = 0; neighbour_i < neighbour_cells.size();
                 neighbour_i++) {
                size_t neighbour_cell = neighbour_cells(neighbour_i);
                if (neighbour_cell < grid_host.num_cells()) {
                    serial_entries.push_back(neighbour_cell);
                }
            }
            serial_row_map.push_back(serial_entries.size());
        }

        Offset num_rows = grid.num_cells();
        Ordinal num_non_zero = serial_entries.size();
        Kokkos::View<Offset*, ExecSpace> row_map("CRS::row_map", num_rows + 1);
        Kokkos::View<Ordinal*, ExecSpace> entries("CRS::column", num_non_zero);

        auto h_row_map = Kokkos::create_mirror_view(row_map);
        auto h_entries = Kokkos::create_mirror_view(entries);
        for (Offset i = 0; i < num_rows + 1; i++) {
            h_row_map(i) = serial_row_map[i];
        }
        for (Ordinal i = 0; i < num_non_zero; i++) {
            h_entries(i) = serial_entries[i];
        }
        Kokkos::deep_copy(row_map, h_row_map);
        Kokkos::deep_copy(entries, h_entries);

        // Create a kernel handle
        using KernelHandle_type = KokkosKernels::Experimental::KokkosKernelsHandle<
            Offset, Ordinal, double, ExecSpace, MemSpace, MemSpace>;

        KernelHandle_type kernel_handle;
        kernel_handle.create_distance2_graph_coloring_handle(
            KokkosGraph::COLORING_D2_DEFAULT);

        KokkosGraph::Experimental::graph_color_distance2(&kernel_handle, num_rows,
                                                         row_map, entries);
        auto colours =
            kernel_handle.get_distance2_graph_coloring_handle()->get_vertex_colors();
        num_colours_ = kernel_handle.get_distance2_graph_coloring_handle()->get_num_colors();

        kernel_handle.destroy_distance2_graph_coloring_handle();
        colours_ = colours;
    }

private:
    Ibis::Array1D<Ordinal> colours_;
    size_t num_colours_;
};

#endif
