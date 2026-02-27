#ifndef ILU_H
#define ILU_H

#include <linear_algrebra/linear_system.h>
#include <linear_algebra/gmres.h>
#include <linear_algebra/crs.h>

class ILU : IterativeLinearSolver {
public:
    ILU(std::shared_ptr<LinearSystem> system, size_t k=0);

    LinearSolveResult solve(Ibis::Vector<Ibis::real>& x);

    void decompose();

private:
    std::shared_ptr<LinearSystem> system_;
    Ibis::CrsMatrix<int, int, double, Ibis::Vector::Layout, class MemSpace>
};

#endif
