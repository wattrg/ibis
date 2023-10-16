import math
flow_state = FlowState(p=101325, T=300.0, vx=3*math.sqrt(1.4*287*300), vy=0.0)
max_time = 5e-3

config.convective_flux = ConvectiveFlux(
    flux_calculator = FluxCalculator.Hanel,
    reconstruction_order = 1
)

config.solver = RungeKutta(
    cfl = 0.5,
    max_step = 100000,
    max_time = max_time,
    plot_every_n_steps = -1,
    plot_frequency = max_time / 10,
    print_frequency = 500
)

config.grid = Block(
    file_name="grid.su2", 
    initial_condition=flow_state,
    boundaries = {
        "inflow": supersonic_inflow(flow_state),
        "outflow": supersonic_outflow(),
        "wall": slip_wall(),
    }
)
