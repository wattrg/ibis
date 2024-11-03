gas_model = IdealGas(species = "air")
gas_state = GasState()
gas_state.T = 300
gas_state.rho = 1.225
gas_model.update_thermo_from_rhoT(gas_state)
flow_state = FlowState(gas=gas_state, vx=1000.0, vy=0.0)

config.convective_flux = ConvectiveFlux(
    flux_calculator=Hanel(),
    reconstruction_order=2
)

config.gas_model = gas_model

config.solver = RungeKutta(
    method="euler",
    cfl=0.5,
    max_step=1,
    max_time=1.0,
    plot_frequency=-0.1,
    plot_every_n_steps=1,
    print_frequency=1
)

config.grid = Block(
    file_name="grid.su2",
    initial_condition=flow_state,
    boundaries={
        "left": supersonic_inflow(flow_state),
        "top": slip_wall(),
        "bottom": slip_wall(),
        "right": supersonic_outflow()
    },
    # motion = RigidBodyTranslation(velocity={"x": 1000.0, "y": 0.0, "z": 0.0})
    motion=ShockFitting(
        boundaries={
            "left": fixed_velocity(Vector3(0.0, 0.0, 0.0)),
            "right": fixed_velocity(Vector3(500.0, 0.0, 0.0)),
            "top": constrained_interpolation(sample_points=["left", "right"], constraint_direction=Vector3(1.0, 0.0, 0.0)),
            "bottom": constrained_interpolation(sample_points=["left", "right"], constraint_direction=Vector3(1.0, 0.0, 0.0))
        }
    )
)
