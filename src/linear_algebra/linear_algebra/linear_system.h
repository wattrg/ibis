#ifndef LINEAR_SYSTEM_H
#define LINEAR_SYSTEM_H

#include <linear_algebra/dense_linear_algebra.h>
#include <simulation/simulation.h>
#include <util/numeric_types.h>

class LinearSystem {
public:
    LinearSystem(){};

    virtual ~LinearSystem() {}

    virtual void matrix_vector_product(Ibis::Vector<Ibis::real>& vec,
                                       Ibis::Vector<Ibis::real>& result) = 0;

    virtual std::unique_ptr<LinearSystem> preconditioner() = 0;

    virtual void eval_rhs() = 0;

    virtual void set_rhs(Ibis::Vector<Ibis::real>& rhs) = 0;

    KOKKOS_INLINE_FUNCTION
    virtual Ibis::real& rhs(const size_t i) const = 0;

    KOKKOS_INLINE_FUNCTION
    virtual Ibis::real& rhs(const size_t cell_i, const size_t cons_i) const = 0;

    virtual Ibis::Vector<Ibis::real>& rhs() = 0;

    virtual size_t num_vars() const = 0;
};

#endif
