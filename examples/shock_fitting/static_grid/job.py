import math
gas_model = IdealGas(species="air")
gas_state = GasState()
gas_state.p = 1.013e3
gas_state.T = 300.0
gas_model.update_thermo_from_pT(gas_state)
v = 3.0 * gas_model.speed_of_sound(gas_state)
inflow = FlowState(gas=gas_state, vx=v)
initial = FlowState(gas=gas_state, vx=v)

config.convective_flux = ConvectiveFlux(
    flux_calculator=Hanel(),
    reconstruction_order=1,
)

config.viscous_flux = ViscousFlux(enabled = False)

config.gas_model = gas_model

# config.solver = RungeKutta(
#     cfl = 0.5,
#     max_step = 500000,
#     max_time = max_time,
#     plot_every_n_steps = -20,
#     plot_frequency = max_time / 10,
#     print_frequency = 1000,
# )

config.solver = SteadyState(
    cfl=ResidualBasedCfl(
        growth_threshold=0.9,
        power=1.2,
        start_cfl=0.5,
        max_cfl=1e6
    ),
    max_steps=10000,
    plot_frequency=100,
    print_frequency=20,
    diagnostics_frequency=1,
    tolerance=1e-8,
    linear_solver=FGmres(
        tolerance=1e-3,
        max_iters=100,
        preconditioner_tolerance=1e-2,
        max_preconditioner_iters=30
    )
)

config.grid = Block(
    file_name="sphere.su2",
    initial_condition=inflow,
    boundaries = {
        "wall": slip_wall(), # wall
        "slip_wall": slip_wall(), # symmetry
        "inflow": supersonic_inflow(inflow), # inflow
        "outflow": supersonic_outflow(),
    }
)
