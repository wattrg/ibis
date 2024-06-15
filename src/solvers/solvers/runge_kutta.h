#ifndef RUNGE_KUTTA_H
#define RUNGE_KUTTA_H

#include <finite_volume/conserved_quantities.h>
#include <gas/flow_state.h>
#include <gas/transport_properties.h>
#include <grid/grid.h>
#include <io/io.h>
#include <solvers/cfl.h>
#include <solvers/solver.h>

#include <memory>

class ButcherTableau {
public:
    ButcherTableau() {}

    ButcherTableau(std::vector<std::vector<double>> a, std::vector<double> b,
                   std::vector<double> c)
        : a_(a), b_(b), c_(c) {
        num_stages_ = b_.size();
    }

    ButcherTableau(json config)
        : ButcherTableau(config.at("a"), config.at("b"), config.at("c")) {}

    double a(size_t i, size_t j);
    double b(size_t i);
    double c(size_t i);
    size_t num_stages();

private:
    std::vector<std::vector<double>> a_;
    std::vector<double> b_;
    std::vector<double> c_;
    size_t num_stages_;
};

class RungeKutta : public Solver {
public:
    RungeKutta(json config, GridBlock<double>& grid, std::string grid_dir,
               std::string flow_dir);

    ~RungeKutta() {}

    int solve();

private:
    // configuration
    double max_time_;
    unsigned int max_step_;
    unsigned int print_frequency_;
    double plot_frequency_;
    double residual_frequency_;
    int residuals_every_n_steps_;
    int plot_every_n_steps_;
    std::unique_ptr<CflSchedule> cfl_;
    double dt_init_;

private:
    // progress
    double time_since_last_plot_;
    double time_since_last_residual_;
    double t_;
    double dt_;
    double stable_dt_;

private:
    // input/output
    FVIO<double> io_;

private:
    // implementation
    int initialise();
    int finalise();
    int take_step();
    void estimate_dt();
    bool print_this_step(unsigned int step);
    bool plot_this_step(unsigned int step);
    int plot_solution(unsigned int step);
    void print_progress(unsigned int step, double wc);
    std::string stop_reason(unsigned int step);
    bool stop_now(unsigned int step);
    int max_step() const { return max_step_; }
    int count_bad_cells() { return fv_.count_bad_cells(flow_, grid_.num_cells()); }

    // this computes the L2 norms of the time derivates evaluated
    // at the beginning of the previous step (essential whatever is in k_[0]).
    // It should be called after taking a step, so the 
    ConservedQuantitiesNorm<double> L2_norms();

    bool residuals_this_step(unsigned int step);
    bool write_residuals(unsigned int step);

private:
    // memory
    FlowStates<double> flow_;
    ConservedQuantities<double> conserved_quantities_;
    std::vector<ConservedQuantities<double>> k_;
    ConservedQuantities<double> k_tmp_;
    FlowStates<double> flow_tmp_;

    // butcher tableau
    ButcherTableau tableau_;

private:
    // spatial discretisation
    GridBlock<double> grid_;
    FiniteVolume<double> fv_;

private:
    IdealGas<double> gas_model_;
    TransportProperties<double> trans_prop_;
};

#endif
