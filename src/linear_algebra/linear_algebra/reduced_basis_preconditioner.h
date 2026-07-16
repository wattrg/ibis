#ifndef REDUCED_BASIS_PRECONDITIONER_H
#define REDUCED_BASIS_PRECONDITIONER_H

#include <linear_algebra/linear_solver.h>
#include <linear_algebra/dense_linear_algebra.h>
#include "util/conserved_quantities.h"

template <class MemModel>
class ReducedBasisPreconditioner : public DirectLinearSolver {
public:
    using MemSpace = Ibis::DefaultMemSpace;
    using HostMemSpace = Ibis::DefaultHostMemSpace;
    using ArrayLayout = Ibis::DefaultArrayLayout;
    using HostArrayLayout = Ibis::DefaultHostArrayLayout;
    using HostExecSpace = Ibis::DefaultHostExecSpace;

public:
    ~ReducedBasisPreconditioner() {};

    ReducedBasisPreconditioner(std::shared_ptr<LinearSystem> system, size_t nb = 2);

    void solve(Ibis::Vector<Ibis::real>& rhs, Ibis::Vector<Ibis::real>& x);

    void update_basis(ConservedQuantities<Ibis::dual>& sol);

private:
    std::shared_ptr<LinearSystem> system_;
    Ibis::Matrix<Ibis::real> W_; // device
    Ibis::Matrix<Ibis::real, HostExecSpace> D_; // host
    Ibis::Matrix<Ibis::real> V_; // device
    Ibis::Matrix<Ibis::real, HostExecSpace> H_; // host
    Ibis::Vector<Ibis::real> z_; // device
    Ibis::Vector<Ibis::real> w_dev_; // device
    Ibis::Vector<Ibis::real, HostExecSpace> w_host_; // host
    Ibis::Vector<Ibis::real> p_dev_; // device
    Ibis::Vector<Ibis::real, HostExecSpace> p_host_; // host
    Ibis::Vector<Ibis::real> d_; // device
    Ibis::Vector<Ibis::real> s_; // device
    
};

#endif
