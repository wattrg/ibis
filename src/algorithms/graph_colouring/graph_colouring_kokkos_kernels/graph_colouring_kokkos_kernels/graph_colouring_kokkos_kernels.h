#ifndef KOKKOS_KERNEL_GRAPH_COLOUR_H
#define KOKKOS_KERNEL_GRAPH_COLOUR_H

#include <graph_colouring/graph_colouring_interface.h>

#include <KokkosGraph_Distance2Color.hpp>

using Scalar = KokkosKernels::default_scalar;
using Ordinal = KokkosKernels::default_lno_t;
using Offset = KokkosKernels::default_size_type;

template <class GridBlock_type>
class GridColourer;

template <class CrsGraphType>
class KokkosKernels_GraphColourer : public GraphColourer<CrsGraphType> {
public:
    ~KokkosKernels_GraphColourer() = default;

    Ibis::Array1D<Ordinal> colours() const { return colours_; }

    int num_colours() const { return num_colours_; }

    void compute_colouring(const CrsGraphType& graph) {
        // using Device = ExecSpace::device_type;
        using MemSpace = CrsGraphType::memory_space;
        using ExecSpace = CrsGraphType::execution_space;

        // Create a kernel handle
        using KernelHandle_type = KokkosKernels::Experimental::KokkosKernelsHandle<
            Offset, Ordinal, double, ExecSpace, MemSpace, MemSpace>;

        KernelHandle_type kernel_handle;
        kernel_handle.create_distance2_graph_coloring_handle(
            KokkosGraph::COLORING_D2_DEFAULT);

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
