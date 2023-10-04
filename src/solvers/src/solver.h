#ifndef SOLVER_H
#define SOLVER_H

#include <fstream>
#include <iostream>
#include <string.h>
#include <nlohmann/json.hpp>
#include "../../finite_volume/src/finite_volume.h"
#include "../../gas/src/flow_state.h"

using json = nlohmann::json;

class Solver {
public:
    Solver(std::string grid_dir, std::string flow_dir);
    int solve();
    virtual ~Solver() {}

protected:
    int write_solution();
    virtual int initialise()=0;
    virtual int finalise()=0;
    virtual int take_step()=0;
    virtual bool print_this_step(unsigned int step)=0;
    virtual bool plot_this_step(unsigned int step)=0;
    virtual int plot_solution(unsigned int step)=0;
    virtual std::string progress_string(unsigned int step)=0;
    virtual bool stop_now(unsigned int step)=0;
    virtual std::string stop_reason(unsigned step)=0;
    virtual int max_step() const = 0;

    unsigned int max_step_ = 0;
    std::string grid_dir_;
    std::string flow_dir_;
};

std::unique_ptr<Solver> make_solver(json config, std::string grid_dir, std::string flow_dir);

template<typename T>
int read_initial_condition(FlowStates<T>& fs, std::string flow_dir);

template<typename T>
int write_flow_solution(const FlowStates<T>& fs, const GridBlock<T>& grid, const std::string flow_dir, int flow_i);


#endif
