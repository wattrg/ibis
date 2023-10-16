import math

mach = 3.0
T = 300
n_flows = 10
n_plots = 10
length = 1.0
vx = mach * math.sqrt(1.4 * 287 * T)
flow_state = FlowState(rho=1.225, T=T, vx=vx, vy=0.0)

config.convective_flux = ConvectiveFlux(
    flux_calculator = FluxCalculator.Hanel,
    reconstruction_order = 1
)

config.solver = RungeKutta(
    cfl = 0.5,
    max_step = 100000,
    max_time = n_flows * length / vx,
    plot_every_n_steps = -1,
    plot_frequency = n_flows / n_plots * length / vx,
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
