#ifndef KOKKOS_KERNEL_TRIANGULAR_SOLVER_H
#define KOKKOS_KERNEL_TRIANGULAR_SOLVER_H

#include <linear_algebra/crs.h>
#include <linear_algebra/dense_linear_algebra.h>
#include <triangular_solver/triangular_solver.h>

#include <KokkosSparse_sptrsv.hpp>

using Scalar = KokkosKernels::default_scalar;
using Ordinal = KokkosKernels::default_lno_t;
using Offset = KokkosKernels::default_size_type;

template <class ExecSpace, class MemSpace, class Layout>
class KokkosKernels_SparseTriangularSolver {
private:
    using KernelHandle_type =
        KokkosKernels::Experimental::KokkosKernelsHandle<Offset, Ordinal, double,
                                                         ExecSpace, MemSpace, MemSpace>;

public:
    KokkosKernels_SparseTriangularSolver() {}

    KokkosKernels_SparseTriangularSolver(
        const Ibis::CrsMatrix<Offset, Ordinal, Scalar>& A,
        const Ibis::TriangularMatrixType type) {
        symbolic_phase(A, type);
    }

    ~KokkosKernels_SparseTriangularSolver() {
        if (handle_.get_sptrsv_handle()) {
            handle_.destroy_sptrsv_handle();
        }
    }

    void symbolic_phase(const Ibis::CrsMatrix<Offset, Ordinal, Scalar>& A,
                        const Ibis::TriangularMatrixType type) {
        int num_rows = A.num_rows();

        // handle_ = KernelHandle_type kernel_handle;
        bool is_lower_tri = (type == Ibis::TriangularMatrixType::LOWER) ? true : false;
        handle_.create_sptrsv_handle(
            KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_TP1, num_rows,
            is_lower_tri);

        KokkosSparse::sptrsv_symbolic(&handle_, A.graph.row_map, A.graph.entries,
                                      A.values);
        Kokkos::fence();
    }

    void solve(const Ibis::CrsMatrix<Offset, Ordinal, Scalar>& A,
               Ibis::Vector<Ibis::real>& b, Ibis::Vector<Ibis::real>& x) {
        KokkosSparse::sptrsv_solve(&handle_, A.graph.row_map, A.graph.entries, A.values,
                                   b.data(), x.data());
        Kokkos::fence();
    }

private:
    KernelHandle_type handle_;
};

#endif
