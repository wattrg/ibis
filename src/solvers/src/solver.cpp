#include <iostream>
#include "solver.h"

int Solver::solve() {
    std::cout << "Begining solve" << std::endl;
    
    for (unsigned int step = 0; step < _max_step; step++) {
        take_step();

        if (print_this_step()) {
            print_progress();
        }

        if (plot_this_step()) {
            plot_solution();
        }

        if (stop()) {
            print_stop_reason();
            break;
        }
    }
}
