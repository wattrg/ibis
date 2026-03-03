#ifndef ILU_H
#define ILU_H

#include <ilu_kokkos_kernels/ilu_kokkos_kernels.h>
#include <linear_algebra/crs.h>
#include <linear_algebra/linear_solver.h>
#include <linear_algebra/linear_system.h>
#include <triangular_solver_kokkos_kernels/triangular_solver_kokkos_kernels.h>
#include <util/types.h>

class DirectPreconditioner : DirectLinearSolver {
    virtual void solve(Ibis::Vector<Ibis::real>& x) = 0;

    virtual void update_preconditioner() = 0;  
};

template <class MemModel>
class ILU : DirectPreconditioner {
public:
    ILU(std::shared_ptr<LinearSystem> system, size_t k = 0);

    void solve(Ibis::Vector<Ibis::real>& x);

    void update_preconditioner();

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

    Ibis::Vector<Ibis::real> temp_vec_;
};

#endif
