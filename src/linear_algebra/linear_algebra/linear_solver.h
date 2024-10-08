#ifndef LINEAR_SOLVER_H
#define LINEAR_SOLVER_H

#include <linear_algebra/linear_system.h>

class LinearSolver {
    void initialise(LinearSystem& linear_system);

    void solve();
};

#endif
