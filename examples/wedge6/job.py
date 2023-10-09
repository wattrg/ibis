flow_state = FlowState(rho=1.225, T=600.0, vx=1000.0, vy=0.0)

config.convective_flux = ConvectiveFlux(
    flux_calculator = FluxCalculator.Hanel,
    reconstruction_order = 1
)

config.solver = RungeKutta(
    cfl = 0.5,
    max_step = 100000,
    max_time = 5 * 0.3 / 1000,
    plot_every_n_steps = -1,
    plot_frequency = 0.3 / 1000,
    print_frequency = 2000
)

config.grid = Block(
    file_name="grid_coarse.su2", 
    initial_condition=flow_state,
    boundaries = {
        "west": supersonic_inflow(flow_state),
        "north": supersonic_outflow(),
        "south": slip_wall(),
        "east": supersonic_outflow()
    }
)
