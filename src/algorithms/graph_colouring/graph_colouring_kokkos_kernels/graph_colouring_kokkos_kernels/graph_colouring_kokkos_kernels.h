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
    Ibis::Array1D<Ordinal> colours() const { return colours_; }

    int num_colours() const { return num_colours_; }

    void compute_colouring(const GridBlock_type& grid) {
        // using Device = ExecSpace::device_type;
        using MemSpace = GridBlock_type::memory_space;
        using ExecSpace = GridBlock_type::execution_space;

        // Create a kernel handle
        using KernelHandle_type = KokkosKernels::Experimental::KokkosKernelsHandle<
            Offset, Ordinal, double, ExecSpace, MemSpace, MemSpace>;

        KernelHandle_type kernel_handle;
        kernel_handle.create_distance2_graph_coloring_handle(
            KokkosGraph::COLORING_D2_DEFAULT);

        auto graph = grid.graph(1);
        KokkosGraph::Experimental::graph_color_distance2(&kernel_handle, graph.num_rows(),
                                                         graph.row_map, graph.entries);
        auto colours =
            kernel_handle.get_distance2_graph_coloring_handle()->get_vertex_colors();
        num_colours_ =
            kernel_handle.get_distance2_graph_coloring_handle()->get_num_colors();

        kernel_handle.destroy_distance2_graph_coloring_handle();
        colours_ = colours;
    }

private:
    Ibis::Array1D<Ordinal> colours_;
    int num_colours_;
};

#endif
