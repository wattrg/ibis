config.convective_flux.flux_calculator = FluxCalculator.Hanel
config.convective_flux.reconstruction_order = 1
config.solver = RungeKutta()
config.solver.cfl = 0.5

flow_state = FlowState(rho=1.225, T=300.0, vx=1000.0)
config.grid = Block(
    file_name="grid.su2", 
    initial_condition=flow_state,
    boundaries = {
        "left": SupersonicInflow(flow_state)
    }
)
