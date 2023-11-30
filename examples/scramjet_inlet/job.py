mach = 5.27
T = 220
p = 2416
n_flows = 3
n_plots = 3
length = 2.3
gas_model = IdealGas(287.0)
gas_state = GasState()
gas_state.p = p
gas_state.T = T
gas_model.update_thermo_from_pT(gas_state)
vx = mach * gas_model.speed_of_sound(gas_state)
flow_state = FlowState(gas=gas_state, vx=vx)

config.convective_flux = ConvectiveFlux(
    flux_calculator = FluxCalculator.Ausmdv,
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
        "symmetry": slip_wall(),
    }
)
