gas_model = IdealGas(R=287.0)
gas_state = GasState()
gas_state.p = 101325
gas_state.T = 300.0
gas_model.update_thermo_from_pT(gas_state)
vx = 3 * gas_model.speed_of_sound(gas_state)
flow_state = FlowState(gas=gas_state, vx=vx)
max_time = 5e-3

config.convective_flux = ConvectiveFlux(
    flux_calculator = Ausmdv(),
    reconstruction_order = 2
)

config.gas_model = gas_model

config.solver = RungeKutta(
    method = "ssp-rk3",
    cfl = 5.0,
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
