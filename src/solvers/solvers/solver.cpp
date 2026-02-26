#include <parallel/parallel.h>
#include <solvers/runge_kutta.h>
#include <solvers/solver.h>
#include <solvers/steady_state.h>
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

#ifdef Ibis_ENABLE_MPI
#include <ibis_mpi/ibis_mpi_conserved_quantities.h>
#include <ibis_mpi/ibis_mpi_dual.h>
#endif

using json = nlohmann::json;

Solver::Solver(std::string grid_dir, std::string flow_dir)
    : grid_dir_(grid_dir), flow_dir_(flow_dir) {}

int Solver::solve() {
    int success = initialise();
    bool is_master = is_master_();
    if (success != 0) {
        spdlog::error("Failed to initialise runge kutta solver");
        return success;
    }
    spdlog::stopwatch sw;
    for (size_t step = 0; step < max_step(); step++) {
        int result = take_step(step);

        if (is_master && residuals_this_step(step)) {
            write_residuals(step, sw.elapsed().count());
        }

        if (result != 0) {
            if (is_master) {
                spdlog::error("step {} failed", step);
            }
            plot_solution(step);
            return 1;
        }

        int bad_cells = count_bad_cells();
        if (bad_cells > 0) {
            if (is_master) {
                spdlog::error("Encountered {} bad cells on step {}", bad_cells, step);
            }
            plot_solution(step);
            return 1;
        }

        if (stop_now(step)) {
            if (is_master) {
                std::string reason = stop_reason(step);
                spdlog::info("STOPPING: {}", reason);
            }
            plot_solution(step);
            break;
        }

        if (is_master && print_this_step(step)) {
            print_progress(step, sw.elapsed().count());
        }

        if (plot_this_step(step)) {
            plot_solution(step);
        }
    }
    if (is_master) {
        spdlog::info("Elapsed Wall Clock: {:.3}s", sw);
    }
    finalise();

    return 0;
}

template <class MemModel>
std::unique_ptr<Solver> make_solver(json config, std::string grid_dir,
                                    std::string flow_dir) {
    size_t grid_id = Ibis::get_world_rank<MemModel>();
    std::filesystem::path base_path = std::filesystem::path(grid_dir);
    std::filesystem::path grid_path = base_path / "0000";
    json solver_config = config.at("solver");
    json grid_config = config.at("grids");
    std::string solver_name = solver_config.at("name");
    if (solver_name == "runge_kutta") {
        GridBlock<MemModel, Ibis::real> grid(grid_path, grid_config[grid_id]);
        return std::unique_ptr<Solver>(
            new RungeKutta<MemModel>(config, std::move(grid), grid_dir, flow_dir));
    } else if (solver_name == "steady_state") {
        GridBlock<MemModel, Ibis::dual> grid(grid_path, grid_config[grid_id]);
        return std::unique_ptr<Solver>(
            new SteadyState<MemModel>(config, std::move(grid), grid_dir, flow_dir));
    } else {
        spdlog::error("Unknown solver {}", solver_name);
        throw new std::runtime_error("Unknown solver");
    }
    return NULL;
}

template std::unique_ptr<Solver> make_solver<SharedMem>(json, std::string, std::string);
template std::unique_ptr<Solver> make_solver<Mpi>(json, std::string, std::string);
