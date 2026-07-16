#ifndef LINEAR_SOLVER_H
#define LINEAR_SOLVER_H

#include <linear_algebra/dense_linear_algebra.h>
#include <linear_algebra/linear_system.h>

class DirectLinearSolver {
public:
    virtual ~DirectLinearSolver() = default;
    virtual void solve(Ibis::Vector<Ibis::real>& rhs, Ibis::Vector<Ibis::real>& x) = 0;
};

// class DirectPreconditioner : public DirectLinearSolver {
// public:
//     virtual ~DirectPreconditioner() = default;

//     virtual void solve(Ibis::Vector<Ibis::real>& rhs, Ibis::Vector<Ibis::real>& x) = 0;

//     virtual void update_preconditioner() = 0;
// };

#endif
