#ifndef KOKKOS_KERNEL_GRAPH_COLOUR_H
#define KOKKOS_KERNEL_GRAPH_COLOUR_H

#include <KokkosSparse_spiluk.hpp>
#include <linear_algrabra/crs.h>

using Scalar = KokkosKernels::default_scalar;
using Ordinal = KokkosKernels::default_lno_t;
using Offset = KokkosKernels::default_size_type;

template <class ExecSpace, class MemSpace>
class KokkosKernels_ILU {
public:
    void symbolic_phase(Ibis::CrsMatrix<Offset, Ordinal, Scalar>& A, int fill_level) {
        fill_level_ = fill_level;
        // Create a kernel handle

        KernelHandle_type kernel_handle;
        handle_ = kernel_handle.create_distance2_graph_coloring_handle();
        int num_rows = A.num_rows();
        int num_entries = A.num_entries();
        handle_.create_spiluk_handle(KokkosSparse::Experimental::SPILUKAlgorithm::SEQLVLSCHD_TP1,
                                     num_rows, num_entries*(fill_level+1),
                                     num_entries*(fill_level+1));

        lno_view_t L_row_map("L_row_map", num_rows + 1);
        lno_nnz_view_t L_entries("L_entries", handle_->get_nnzL());
        scalar_view_t  L_values ("L_values",  handle_->get_nnzL());
        lno_view_t     U_row_map("U_row_map", N + 1);
        lno_nnz_view_t U_entries("U_entries", handle_->get_nnzU());
        scalar_view_t  U_values ("U_values",  handle_->get_nnzU());
    }

    // void numeric_phase(Ibis::CrsMatrix<Offset, Ordinal, Scalar>& A, )

private:
    using KernelHandle_type = KokkosKernels::Experimental::KokkosKernelsHandle<
        Offset, Ordinal, double, ExecSpace, MemSpace, MemSpace>;

    KernelHandle_type::SPILUKHandleType handle_;

    Ibis::CrsMatrix<Offset, Ordinal, Scalar> L_;
    Ibis::CrsMatrix<Offset, Ordinal, Scalar> U_;

    int fill_level_;
};

#endif
