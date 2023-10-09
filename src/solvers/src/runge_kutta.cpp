#include <spdlog/spdlog.h>
#include "runge_kutta.h"
#include "solver.h"
#include "../../finite_volume/src/primative_conserved_conversion.h"


RungeKutta::RungeKutta(json config, GridBlock<double> grid, std::string grid_dir, std::string flow_dir) 
    : Solver(grid_dir, flow_dir)
{
    // configuration
    json solver_config = config.at("solver");
    max_time_ = solver_config.at("max_time");
    max_step_ = solver_config.at("max_step");
    print_frequency_ = solver_config.at("print_frequency");
    plot_frequency_ = solver_config.at("plot_frequency");
    plot_every_n_steps_ = solver_config.at("plot_every_n_steps");
    cfl_ = solver_config.at("cfl"); 

    // memory
    grid_ = grid;
    int number_cells = grid_.num_total_cells();
    int dim = grid_.dim();
    flow_ = FlowStates<double>(number_cells);
    conserved_quantities_ = ConservedQuantities<double>(number_cells, dim);
    dUdt_ = ConservedQuantities<double>(number_cells, dim);
    fv_ = FiniteVolume<double>(grid_, config);

    // progress
    time_since_last_plot_ = 0.0;
    t_ = 0.0;

    // input/output
    io_ = FVIO<double>();
}

int RungeKutta::initialise() {
    // int ic_result = read_initial_condition(flow_, flow_dir_, grid_.num_cells());
    int ic_result = io_.read(flow_, grid_, 0);
    int conversion_result = primatives_to_conserved(conserved_quantities_, flow_);
    return ic_result + conversion_result;
}

int RungeKutta::finalise() {
    return 0;
}

int RungeKutta::take_step() {
    fv_.compute_dudt(flow_, grid_, dUdt_);
    double full_dt = cfl_ * fv_.estimate_dt(flow_, grid_);
    dt_ = Kokkos::min(full_dt, max_time_ - t_);
    conserved_quantities_.apply_time_derivative(dUdt_, dt_);
    conserved_to_primatives(conserved_quantities_, flow_);

    t_ += dt_;
    time_since_last_plot_ += dt_;
    return 0;
}
bool RungeKutta::print_this_step(unsigned int step) {
    if (step != 0 && step % print_frequency_ == 0) return true;
    return false;
}

bool RungeKutta::plot_this_step(unsigned int step) {
    if (plot_every_n_steps_ > 0 && step != 0 && step % plot_every_n_steps_ == 0) return true;
    if (plot_frequency_ > 0 && time_since_last_plot_ > plot_frequency_) return true;
    return false;
}

int RungeKutta::plot_solution(unsigned int step) {
    // int result =  write_flow_solution<double>(flow_, grid_, flow_dir_, n_solutions_);
    int result = io_.write(flow_, grid_, t_);
    time_since_last_plot_ = 0.0;
    spdlog::info("  written flow solution: step {}, time {:.6e}", step, t_);
    return result;
}

void RungeKutta::print_progress(unsigned int step) {
    spdlog::info("  step: {:>8}, t = {:.6e}, dt = {:.6e}", step, t_, dt_);
}
std::string RungeKutta::stop_reason(unsigned int step) {
    if (t_ >= max_time_ - 1e-15) return "reached max time"; // subtract small amount to avoid round off error
    if (step >= max_step_-1) return "reached max step";
    return "Shouldn't reach here";
}
bool RungeKutta::stop_now(unsigned int step) {
    if (step >= max_step_-1) return true;
    if (t_ >= max_time_) return true;
    return false;
}

int RungeKutta::count_bad_cells(){
    int num_cells = grid_.num_cells();
    int n_bad_cells = 0;
    Kokkos::parallel_reduce("RungeKutta::check_state", num_cells, KOKKOS_LAMBDA(const int cell_i, int& n_bad_cells_utd){
        if (flow_.gas.temp(cell_i) < 0.0 || flow_.gas.rho(cell_i) < 0.0) {
            n_bad_cells_utd += 1;
        }
    }, n_bad_cells);
    return n_bad_cells;
}
