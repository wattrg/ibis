gas_model = get_gas_model("ideal_air")
gas_state = GasState()
gas_state.T = 300
gas_state.rho = 1.225
gas_model.update_thermo_from_rhoT(gas_state)
flow_state = FlowState(gas=gas_state, vx=1000.0, vy=500.0)

config.convective_flux = ConvectiveFlux(
    flux_calculator = FluxCalculator.Hanel,
    reconstruction_order = 2
)

config.solver = RungeKutta(
    cfl = 0.5,
    max_step = 100000,
    max_time = 1.0,
    plot_frequency = 0.1,
    print_frequency = 200
)

config.grid = Block(
    file_name="grid.su2", 
    initial_condition=flow_state,
    boundaries = {
        "left": supersonic_inflow(flow_state),
        "top": supersonic_outflow(),
        "bottom": supersonic_inflow(flow_state),
        "right": supersonic_outflow()
    }
)
