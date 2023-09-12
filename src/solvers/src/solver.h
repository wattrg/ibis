#ifndef SOLVER_H
#define SOLVER_H

#include <iostream>
#include <string.h>
#include <nlohmann/json.hpp>
#include "../../finite_volume/src/finite_volume.h"

using json = nlohmann::json;

class Solver {
public:
    int solve();
    virtual ~Solver(){}

protected:
    virtual int initialise()=0;
    virtual int finalise()=0;
    virtual int take_step()=0;
    virtual bool print_this_step()=0;
    virtual bool plot_this_step()=0;
    virtual int plot_solution()=0;
    virtual int print_progress()=0;
    virtual bool stop_now()=0;
    virtual void print_stop_reason()=0;

private:
    unsigned int max_step_ = 0;
};

Solver * make_solver(json solver_config, GridBlock<double> grid);

#endif
