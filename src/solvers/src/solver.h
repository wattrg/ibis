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
    virtual ~Solver(){}

protected:
    virtual int initialise()=0;
    virtual int finalise()=0;
    virtual int take_step()=0;
    virtual bool print_this_step(unsigned int step)=0;
    virtual bool plot_this_step(unsigned int step)=0;
    virtual int plot_solution(unsigned int step)=0;
    virtual int print_progress(unsigned int step)=0;
    virtual bool stop_now(unsigned int step)=0;
    virtual std::string stop_reason(unsigned step)=0;

    unsigned int max_step_ = 0;
    std::string grid_dir_;
    std::string flow_dir_;
};

Solver * make_solver(json config, std::string grid_dir, std::string flow_dir);

template<typename T>
int read_initial_condition(FlowStates<T> fs, std::string flow_dir) {
    std::string line;
    std::ifstream temp(flow_dir + "/0000/T");
    int cell_i = 0;
    while (getline(temp, line)) {
        fs.gas.temp(cell_i) = stoi(line);
        cell_i++; 
    }
    temp.close();

    std::ifstream pressure(flow_dir + "/0000/p");
    cell_i = 0;
    while (getline(temp, line)){
        fs.gas.pressure(cell_i) = stoi(line);
        cell_i++;
    }
    pressure.close();

    std::ifstream vx(flow_dir + "/0000/vx");
    cell_i = 0;
    while (getline(temp, line)){
        fs.vel.x(cell_i) = stoi(line);
        cell_i++;
    }
    vx.close();

    std::ifstream vy(flow_dir + "/0000/vy");
    cell_i = 0;
    while (getline(temp, line)){
        fs.vel.y(cell_i) = stoi(line);
        cell_i++;
    }
    vy.close();
    return 0;
}

#endif
