---
menubar: reference_manual_menu
---
# Runge Kutta
The Runge Kutta solver is an explicit solver.
It is configured by setting `config.solver` in `job.py` to an instance of `RungeKutta`.
For example:
```
config.solver = RungeKutta(
    cfl = 0.5,
    max_step = 10000,
    max_time = 1e-3,
    plot_every_n_steps = -1,
    plot_frequency = 1e-4,
    print_frequency = 500
)
```

Each option is optional, default values will be used if they are not provided.

Each option is described below.

## cfl
The cfl value to dictate how big of a step to take

> Type: `float`\
> Default: `0.5`

## max_step
The maximum number of steps to take.
This can be helps stop a job from spinning its wheels going nowhere.

> Type: `int`\
> Default: `100`

## print_frequency
The number of steps between printing the progress of the simulation to the terminal

> Type: `int`\
> Default: `20`

## max_time
The maximum time to simulate.

> Type: `float`\
> Default: `1e-3`

## plot_every_n_steps
Write a snapshot at intervals a fixed number of steps apart.
This can be useful for debugging a simulation that is struggling to start.
Setting it to `-1` disables this behaviour.

> Type: `int`\
> Default: `-1`

## plot_frequency
Write a snapshot at fixed intervals of simulation time.

> Type: `float`\
> Default: `1e-3`
