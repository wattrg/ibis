inflow = FlowState(rho=1.225, T=400.0, vx=1000.0, vy=0.0)
initial = FlowState(rho=1.225, T=300.0, vx=0.0)

config.convective_flux = ConvectiveFlux(
    flux_calculator = FluxCalculator.Hanel,
    reconstruction_order = 1
)

config.solver = RungeKutta(
    cfl = 0.5,
    max_step = 100,
    max_time = 1.0,
    plot_frequency = -1,
    plot_every_n_steps = 1,
    print_frequency = 1
)

config.grid = Block(
    file_name="grid.su2", 
    initial_condition=initial,
    boundaries = {
        "left": supersonic_inflow(inflow),
        "top": slip_wall(),
        "bottom": slip_wall(),
        "right": supersonic_outflow()
    }
)
