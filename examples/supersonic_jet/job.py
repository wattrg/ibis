gas_model = IdealGas(R = 287.0)
inflow_gs = GasState()
inflow_gs.p = 101325 
inflow_gs.T = 600.0
gas_model.update_thermo_from_pT(inflow_gs)
vx = 2.0 * gas_model.speed_of_sound(inflow_gs)
inflow = FlowState(gas=inflow_gs, vx=vx)


initial_gs = GasState()
initial_gs.p = 101325/2
initial_gs.T = 300
gas_model.update_thermo_from_pT(initial_gs)
initial = FlowState(gas=initial_gs, vx=0)
max_time = 5 * 1.0 / vx

config.convective_flux = ConvectiveFlux(
    flux_calculator = Ausmdv(),
    reconstruction_order = 2
)

config.viscous_flux = ViscousFlux(enabled = False)

config.gas_model = gas_model

config.solver = RungeKutta(
    method="ssp-rk3",
    cfl = 4.0,
    max_step = 10000000,
    max_time = max_time,
    plot_every_n_steps = -1,
    plot_frequency = max_time / 10,
    print_frequency = 1000
)

config.grid = Block(
    file_name="grid.su2", 
    initial_condition=initial,
    boundaries = {
        "inflow": supersonic_inflow(inflow),
        "outflow": supersonic_outflow(),
        "wall": adiabatic_no_slip_wall(),
    }
)
