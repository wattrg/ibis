import math
from parameters import mach_1

mach = mach_1
T = 300
max_time = 0.002
n_plots = 1
length = 1.0
gas_model = IdealGas(R=287)
gas_state = GasState()
gas_state.rho = 1.225
gas_state.T = T
gas_model.update_thermo_from_rhoT(gas_state)
vx = mach * gas_model.speed_of_sound(gas_state)
flow_state = FlowState(gas=gas_state, vx=vx)

config.convective_flux = ConvectiveFlux(
    flux_calculator = Ausmdv(),
    reconstruction_order = 2,
    limiter = Unlimited()
)

config.gas_model = gas_model

config.solver = RungeKutta(
    method="ssp-rk3",
    cfl = 2.0,
    max_step = 100000,
    max_time = max_time,
    plot_every_n_steps = -1,
    plot_frequency = max_time / n_plots,
    print_frequency = 500,
    dt_init = 1e-9
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
