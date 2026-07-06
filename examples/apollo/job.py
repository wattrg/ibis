import math

gas_model = IdealGas(species="air")
gas_state = GasState()
gas_state.p = 500
gas_state.T = 300.0
gas_model.update_thermo_from_pT(gas_state)
v = 5 * gas_model.speed_of_sound(gas_state)
aoa = math.radians(20)
inflow = FlowState(gas=gas_state, vx=v * math.cos(aoa), vy=v * math.sin(aoa), vz=0)
initial = FlowState(gas=gas_state, vx=v * math.cos(aoa), vy=v * math.sin(aoa), vz=0)

config.convective_flux = ConvectiveFlux(
    flux_calculator=Hanel(),
    reconstruction_order=LinearResidualBasedHighOrderBlending(
        start_blending_residual=0.1, stop_blending_residual=0.01
    ),
    limiter=Venkat(),
)

config.viscous_flux = ViscousFlux(enabled=False)

config.gas_model = gas_model

config.solver = SteadyState(
    cfl=ResidualBasedCfl(growth_threshold=0.1, power=1.0, start_cfl=0.5),
    max_steps=10000,
    plot_frequency=100,
    print_frequency=10,
    diagnostics_frequency=1,
    tolerance=1e-6,
    min_relaxation_factor=1e-4,
    linear_solver=Gmres(
        tol=1e-2,
        max_iters=50,
    ),
)

config.grid = Block(
    file_name="grid.su2",
    initial_condition=initial,
    boundaries={
        # "capsule": slip_wall(), # wall
        "capsule": fixed_temperature_no_slip_wall(temperature=1000),
        "inflow": supersonic_inflow(inflow),  # inflow,
        "outflow": supersonic_outflow(),
    },
)
