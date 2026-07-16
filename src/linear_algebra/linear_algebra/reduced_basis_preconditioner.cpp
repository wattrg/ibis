#include <linear_algebra/reduced_basis_preconditioner.h>

template <typename MemModel>
ReducedBasisPreconditioner<MemModel>::ReducedBasisPreconditioner(std::shared_ptr<LinearSystem> system,
                                                                 size_t nb = 2) {
    system_ = system;
    size_t n_vars = system_->num_vars();
    W_ = Ibis::Matrix<Ibis::real>("RBP::W", n_vars, nb);
    D_ = Ibis::Matrix<Ibis::real, HostExecSpace>("RBP::D", nb, nb);
    V_ = Ibis::Matrix<Ibis::real>("RBP::V", n_vars, nb);
    H_ = Ibis::Matrix<Ibis::real, HostExecSpace>("RBP::H", nb, nb);
    w_dev_ = Ibis::Vector<Ibis::real>("RBP::w", nb);
    w_host_ = w_dev_.host_mirror();
    p_dev_ = Ibis::Vector<Ibis::real>("RBP::p", nb);
    p_host_ = p_dev_.host_mirror();
    d_ = Ibis::Vector<Ibis::real>("RBP::d", n_vars);
    z_ = Ibis::Vector<Ibis::real>("RBP::z", n_vars);
    s_ = Ibis::Vector<Ibis::real>("RBP::s", n_vars);
}
