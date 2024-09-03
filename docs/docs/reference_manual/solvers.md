---
toc: true
title: Solvers
---
## Runge Kutta
The Runge Kutta solver is an explicit solver.
It is configured by setting `config.solver` in `job.py` to an instance of `RungeKutta`.
For example:
```
config.solver = RungeKutta(
    method="ssp-rk3",
    cfl = 0.5,
    max_step = 10000,
    max_time = 1e-3,
    plot_every_n_steps = -1,
    plot_frequency = 1e-4,
    print_frequency = 500,
    residual_frequency = 1e-5,
    residuals_every_n_steps = 20,
)
```

Each option is optional, default values will be used if they are not provided.

Each option is described below.

### method
The specific Runge-Kutta scheme to use.
The options are:
  + `euler`
  + `midpoint`
  + `rk3`
  + `ssp-rk3`
  + `rk4`

> Type: `str`\
> Default: `ssp-rk3`

### butcher_tableau
Instead of using one of the existing Runge-Kutta methods, you may provide the Butcher tableau for any explicit Runge Kutta method.
The tableau is specified as a dictionary of the `b`, `b`, and `c` values of the Butcher tableau.
Only the `a` values below the diagonal should are included, and the first `c` value, which is always 0 for explicit Runge-Kutta methods is not included either.
For example, the classic RK4 method can be specified as:
```
butcher_tableau = {
  "a": [[0.5], [0, 0.5], [0, 0, 1]],
  "b": [1/6, 1/3, 1/3, 1/6],
  "c": [0.5, 0.5, 1.0]
}
```

It is an error to provide both `method` and `butcher_tableau` to the `RungeKutta` constructor.

### cfl
The cfl value to dictate how big of a step to take

> Type: `float`\
> Default: `0.5`

### max_step
The maximum number of steps to take.
This can be helps stop a job from spinning its wheels going nowhere.

> Type: `int`\
> Default: `100`

### print_frequency
The number of steps between printing the progress of the simulation to the terminal

> Type: `float`\
> Default: `1e-5`

### residuals_every_n_steps
The number of steps between writing norms of the residuals.
Setting to -1 disables this.

> Type: `int`\
> Default: `20`

### residual_frequency
The time interval between writing norms of residuals to file
Setting to -1 disables this.

> Type: `float`\
> Default: `1e-5`

### max_time
The maximum time to simulate.

> Type: `float`\
> Default: `1e-3`

### plot_every_n_steps
Write a snapshot at intervals a fixed number of steps apart.
This can be useful for debugging a simulation that is struggling to start.
Setting it to `-1` disables this behaviour.

> Type: `int`\
> Default: `-1`

### plot_frequency
Write a snapshot at fixed intervals of simulation time.

> Type: `float`\
> Default: `1e-3`

## Steady State
The steady-state solver uses the Jacobian-Free Newton-Krylov method to accelerate convergence to steady-state.
It is configured by setting `config.solver` in `job.py` to an instance of `SteadyState`.
For example:
```
config.solver = SteadyState(
  cfl=ResidualBasedCfl(growth_threshold=0.9, power=0.85, start_cfl=0.5),
  max_steps=1000,
  print_frequency=20,
  plot_frequency=100,
  diagnostics_frequency=1,
  tolerance=1e-10,
  linear_solver=Gmres(tol=1e-4, max_iters=50)
)
```

Each option is optional, default values will be used if they are not provided.

Each option is described below.

### cfl
The cfl to use for the simulation.

> Type: `float` | [`CflSchedule`](cfl_schedules)
> Default: 0.5

### max_steps
The maximum number of steps to take

> Type: `int`\
> Default 1000

### print_frequency
The number of steps between printing progress to the screen

> Type: `int`\
> Default: 10

### plot_frequency
The number of steps between writing the flow solution to disk

> Type: `int`\
> Default: 10

### diagnostics_frequency
The number of steps between writing diagnostics.
Diagnostics are written in the `log` folder.

> Type: `int`\
> Default: 1

### tolerance
The drop in the relative global residual required for convergence

> Type: `float`\
> Default: 1e-5

### linear_solver
The linear to use for each non-linear step

> Type: [`LinearSolver`](linear_solvers)
