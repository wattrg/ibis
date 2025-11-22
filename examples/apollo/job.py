import math
gas_model = IdealGas(species="air")
gas_state = GasState()
gas_state.p = 500
gas_state.T = 300.0
gas_model.update_thermo_from_pT(gas_state)
v = 5 * gas_model.speed_of_sound(gas_state)
aoa = math.radians(20)
inflow = FlowState(gas=gas_state, vx = v * math.cos(aoa), vy=v*math.sin(aoa), vz=0)
initial = FlowState(gas=gas_state, vx = v * math.cos(aoa), vy=v * math.sin(aoa), vz=0)

config.convective_flux = ConvectiveFlux(
    flux_calculator=Hanel(),
    reconstruction_order=1,
    # limiter = Unlimited(),
)

config.viscous_flux = ViscousFlux(enabled = True)

config.gas_model = gas_model

config.solver = SteadyState(
    cfl=ResidualBasedCfl(growth_threshold=0.99, power=1.0, start_cfl=1.0),
    max_steps=10000,
    plot_frequency=1000,
    print_frequency=100,
    diagnostics_frequency=1,
    tolerance=1e-6,
    linear_solver=FGmres(
        tolerance=1e-1,
        max_iters=100,
        preconditioner_tolerance=1e-2,
        max_preconditioner_iters=30
    )
)

config.grid = Block(
    file_name="grid.su2",
    initial_condition=initial,
    boundaries = {
        # "capsule": slip_wall(), # wall
        "capsule": fixed_temperature_no_slip_wall(temperature=1000),
        "inflow": supersonic_inflow(inflow), # inflow,
        "outflow": supersonic_outflow(),
    }
)
