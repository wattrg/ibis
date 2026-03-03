#ifndef ILU_H
#define ILU_H

#include <ilu_kokkos_kernels/ilu_kokkos_kernels.h>
#include <linear_algebra/crs.h>
#include <linear_algebra/gmres.h>
#include <linear_algebra/linear_system.h>
#include <triangular_solver_kokkos_kernels/triangular_solver_kokkos_kernels.h>

#include "util/types.h"

template <class MemModel, typename T>
class ILU : IterativeLinearSolver {
public:
    ILU(GridBlock<MemModel, T> grid, std::shared_ptr<LinearSystem> system, size_t k = 0);

    LinearSolveResult solve(Ibis::Vector<Ibis::real>& x);

    void decompose();

private:
    std::shared_ptr<LinearSystem> system_;
    Ibis::CrsMatrix<int, int, double, Ibis::Vector<Ibis::real>::layout,
                    Ibis::Vector<Ibis::real>::mem_space>
        matrix_;

    using ilu_handle_type =
        KokkosKernels_ILU<Ibis::DefaultExecSpace, Ibis::DefaultMemSpace,
                          Ibis::DefaultArrayLayout>;
    ilu_handle_type ilu_handle_;

    using triangular_solver_handle_type = KokkosKernels_SparseTriangularSolver<
        Ibis::DefaultExecSpace, Ibis::DefaultMemSpace, Ibis::DefaultArrayLayout>;
    triangular_solver_handle_type lower_triangular_solve_handle_;
    triangular_solver_handle_type upper_triangular_solve_handle_;
};

#endif
