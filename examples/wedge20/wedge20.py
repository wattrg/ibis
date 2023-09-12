config.convective_flux.flux_calculator = FluxCalculator.Hanel
config.convective_flux.reconstruction_order = 1

config.solver = RungeKutta()
config.solver.cfl = 0.5

config.grid.add_block("grid.su2")
