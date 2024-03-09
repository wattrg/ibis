#ifndef SOLVER_H
#define SOLVER_H

#include <finite_volume/finite_volume.h>
#include <gas/flow_state.h>
#include <string.h>

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class Solver {
public:
    Solver(std::string grid_dir, std::string flow_dir);
    int solve();
    virtual ~Solver() {}

protected:
    int write_solution();
    virtual int initialise() = 0;
    virtual int finalise() = 0;
    virtual int take_step() = 0;
    virtual bool print_this_step(unsigned int step) = 0;
    virtual bool plot_this_step(unsigned int step) = 0;
    virtual int plot_solution(unsigned int step) = 0;
    virtual void print_progress(unsigned int step, double wc) = 0;
    virtual bool stop_now(unsigned int step) = 0;
    virtual std::string stop_reason(unsigned step) = 0;
    virtual int max_step() const = 0;

    virtual int count_bad_cells() = 0;

    unsigned int max_step_ = 0;
    std::string grid_dir_;
    std::string flow_dir_;
};

std::unique_ptr<Solver> make_solver(json config, std::string grid_dir,
                                    std::string flow_dir);

template <typename T>
int read_initial_condition(FlowStates<T>& fs, std::string flow_dir, int num_cells);

template <typename T>
int write_flow_solution(const FlowStates<T>& fs, const GridBlock<T>& grid,
                        const std::string flow_dir, int flow_i);

#endif
