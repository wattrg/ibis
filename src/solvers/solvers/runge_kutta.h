#ifndef RUNGE_KUTTA_H
#define RUNGE_KUTTA_H

#include <finite_volume/conserved_quantities.h>
#include <gas/flow_state.h>
#include <gas/transport_properties.h>
#include <grid/grid.h>
#include <io/io.h>
#include <solvers/solver.h>

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
    int plot_every_n_steps_;
    double cfl_;

private:
    // progress
    double time_since_last_plot_;
    double t_;
    double dt_;

private:
    // input/output
    FVIO<double> io_;

private:
    // implementation
    int initialise();
    int finalise();
    int take_step();
    bool print_this_step(unsigned int step);
    bool plot_this_step(unsigned int step);
    int plot_solution(unsigned int step);
    void print_progress(unsigned int step, double wc);
    std::string stop_reason(unsigned int step);
    bool stop_now(unsigned int step);
    int max_step() const { return max_step_; }
    int count_bad_cells() {
        return fv_.count_bad_cells(flow_, grid_.num_cells());
    }

private:
    // memory
    FlowStates<double> flow_;
    ConservedQuantities<double> conserved_quantities_;
    ConservedQuantities<double> dUdt_;

private:
    // spatial discretisation
    GridBlock<double> grid_;
    FiniteVolume<double> fv_;

private:
    IdealGas<double> gas_model_;
    TransportProperties<double> trans_prop_;
};

#endif
