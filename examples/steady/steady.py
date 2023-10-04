config.convective_flux.flux_calculator = FluxCalculator.Hanel
config.convective_flux.reconstruction_order = 1
config.solver = RungeKutta()
config.solver.cfl = 0.5
config.solver.max_step = 1
config.solver.max_time = 1
config.solver.plot_every_n_steps = 20
config.solver.plot_frequency = 10000

flow_state = FlowState(rho=1.225, T=300.0, vx=1000.0)
config.grid = Block(
    file_name="grid.su2", 
    initial_condition=flow_state,
    boundaries = {
        "inflow": SupersonicInflow(flow_state),
        "slip_wall_top": SupersonicInflow(flow_state),
        "slip_wall_bottom": SupersonicInflow(flow_state),
        "outflow": SupersonicInflow(flow_state)
    }
)
