import math
gas_model = IdealGas(species="air")
gas_state = GasState()
gas_state.p = 1.013e3
gas_state.T = 300.0
gas_model.update_thermo_from_pT(gas_state)
v = 3.0 * gas_model.speed_of_sound(gas_state)
inflow = FlowState(gas=gas_state, vx=v)
initial = FlowState(gas=gas_state, vx=v)
max_time = 300 * 6.6e-3 / v

config.convective_flux = ConvectiveFlux(
    flux_calculator=Hanel(),
    reconstruction_order=1,
)

config.viscous_flux = ViscousFlux(enabled = False)

config.gas_model = gas_model

# config.solver = RungeKutta(
#     method="midpoint",
#     cfl = 0.4,
#     max_step = 10000000,
#     max_time = max_time,
#     # plot_every_n_steps = 1,
#     plot_frequency = max_time / 50,
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
    file_name="cylinder_eilmer.su2",
    initial_condition=inflow,
    boundaries = {
        "wall": slip_wall(), # wall
        "symmetry": slip_wall(), # symmetry
        "inflow": supersonic_inflow(inflow), # inflow
        "outflow": supersonic_outflow(),
    },
    motion=ShockFitting(
        boundaries={
            "wall": fixed_velocity(Vector3(0.0)),
            "symmetry": constrained_interpolation(
                power=1.5,
                sample_points=["wall", "inflow"],
                constraint=RadialConstraint(Vector3(0.0, 0.0))
            ),
            # "inflow": constrained_interpolation(
            #     sample_points=["shock"],
            #     power=1,
            #     constraint=RadialConstraint(Vector3(0.0, 0.0))
            # ),
            "inflow": shock_fit(
                constraint=RadialConstraint(centre=Vector3(0.0, 0.0, 0.0)),
                scale=0.009,
                shock_detection_threshold=0.3,
            ),
            "outflow": constrained_interpolation(
                sample_points=["wall", "inflow"],
                constraint=RadialConstraint(Vector3(0.0, 0.0)),
                power=1.5,
            ),
            # "shock": shock_fit(
            #     constraint=RadialConstraint(centre=Vector3(0.0, 0.0, 0.0)),
            #     scale = 0.001
            # )
        }
    )
)
