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

    ReducedBasisPreconditioner(std::shared_ptr<LinearSystem> system, int nb);

    void solve(Ibis::Vector<Ibis::real>& rhs, Ibis::Vector<Ibis::real>& x);

    void initialise(ConservedQuantities<Ibis::dual>& sol);

    void update_basis(Ibis::Vector<Ibis::real>& sol);

private:
    int nb_;
    int nb_max_;
    size_t n_vars_;
    std::shared_ptr<LinearSystem> system_;
    Ibis::Matrix<Ibis::real> W_;
    Ibis::Matrix<Ibis::real> D_dev_;
    Ibis::Matrix<Ibis::real> V_;
    Ibis::Matrix<Ibis::real> H_;
    Ibis::Matrix<Ibis::real> H_inv_;
    Ibis::Vector<Ibis::real> z_;
    Ibis::Vector<Ibis::real> w_;
    Ibis::Vector<Ibis::real> p_;
    Ibis::Vector<Ibis::real> d_;
    Ibis::Vector<Ibis::real> s_;
    Ibis::Vector<Ibis::real> vec_tmp_;
    Ibis::Vector<Ibis::real> res_tmp_;
    Ibis::Vector<Ibis::real> M_;

    // bool initialised_;
};

#endif
