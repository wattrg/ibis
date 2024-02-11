gas_model = IdealGas(R = 287.0)
gas_state = GasState()
gas_state.p = 101325
gas_state.T = 300.0
gas_model.update_thermo_from_pT(gas_state)
vx = 2.0 * gas_model.speed_of_sound(gas_state)
inflow = FlowState(gas=gas_state, vx=vx)
initial = FlowState(gas=gas_state, vx=0)
max_time = 2 * 0.0005 / vx

config.convective_flux = ConvectiveFlux(
    flux_calculator = FluxCalculator.Ausmdv,
    reconstruction_order = 2
)

config.viscous_flux = ViscousFlux(enabled = True)

config.gas_model = gas_model

config.solver = RungeKutta(
    cfl = 0.5,
    max_step = 10000000,
    max_time = max_time,
    plot_every_n_steps = -1,
    plot_frequency = max_time / 10,
    print_frequency = 10000
)

config.grid = Block(
    file_name="grid.su2", 
    initial_condition=inflow,
    boundaries = {
        "inflow": supersonic_inflow(inflow),
        "outflow": supersonic_outflow(),
        "wall": adiabatic_no_slip_wall(),
    }
)
