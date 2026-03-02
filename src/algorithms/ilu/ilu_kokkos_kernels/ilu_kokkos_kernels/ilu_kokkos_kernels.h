#ifndef KOKKOS_KERNEL_GRAPH_COLOUR_H
#define KOKKOS_KERNEL_GRAPH_COLOUR_H

#include <KokkosSparse_spiluk.hpp>
#include <linear_algebra/crs.h>

using Scalar = KokkosKernels::default_scalar;
using Ordinal = KokkosKernels::default_lno_t;
using Offset = KokkosKernels::default_size_type;

template <class ExecSpace, class MemSpace, class Layout>
class KokkosKernels_ILU {
private:
    using KernelHandle_type = KokkosKernels::Experimental::KokkosKernelsHandle<
        Offset, Ordinal, double, ExecSpace, MemSpace, MemSpace>;

public:
    KokkosKernels_ILU()=delete;

    KokkosKernels_ILU(Ibis::CrsMatrix<Offset, Ordinal, Scalar>& A, int fill_level) {
        symbolic_phase(A, fill_level);
    }
    
    ~KokkosKernels_ILU() {
        handle_.destroy_spiluk_handle();
    }

    void symbolic_phase(Ibis::CrsMatrix<Offset, Ordinal, Scalar>& A, int fill_level) {
        fill_level_ = fill_level;

        int num_rows = A.num_rows();
        int num_entries = A.num_entries();

        // handle_ = KernelHandle_type kernel_handle;
        handle_.create_spiluk_handle(KokkosSparse::Experimental::SPILUKAlgorithm::SEQLVLSCHD_TP1,
                                     num_rows, num_entries*(fill_level+1),
                                     num_entries*(fill_level+1));
        auto spiluk_handle = handle_.get_spiluk_handle();

        L.row_map = Ibis::Array1D<Offset, Layout, MemSpace>("L_row_map", num_rows + 1);
        L.entries = Ibis::Array1D<Ordinal, Layout, MemSpace>("L_entries", spiluk_handle->get_nnzL());
        L.values = Ibis::Array1D<Scalar, Layout, MemSpace>("L_values",  spiluk_handle->get_nnzL());
        U.row_map = Ibis::Array1D<Offset, Layout, MemSpace>("U_row_map", num_rows + 1);
        U.entries = Ibis::Array1D<Ordinal, Layout, MemSpace>("U_entries", spiluk_handle->get_nnzU());
        U.values = Ibis::Array1D<Scalar, Layout, MemSpace>("U_values",  spiluk_handle->get_nnzU());

        KokkosSparse::spiluk_symbolic(&handle_, fill_level_, A.row_map, A.entries,
                                                L.row_map, L.entries, U.row_map, U.entries);
    }

    void numeric_phase(Ibis::CrsMatrix<Offset, Ordinal, Scalar>& A) {
        KokkosSparse::spiluk_numeric(&handle_, fill_level_,
                                     A.row_map, A.entries, A.values,
                                     L.row_map, L.entries, L.values, U.row_map,
                                     U.entries, U.values );
    }

private:
    KernelHandle_type handle_;

public:
    Ibis::CrsMatrix<Offset, Ordinal, Scalar> L;
    Ibis::CrsMatrix<Offset, Ordinal, Scalar> U;

    int fill_level_;
};

#endif
