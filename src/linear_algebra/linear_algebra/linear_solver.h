#ifndef LINEAR_SOLVER_H
#define LINEAR_SOLVER_H

#include <linear_algebra/dense_linear_algebra.h>
#include <linear_algebra/linear_system.h>

class DirectLinearSolver {
    virtual void solve(Ibis::Vector<Ibis::real>& x) = 0;
};

#endif
