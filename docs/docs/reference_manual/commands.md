---
title: Commands
toc: true
---
`ibis` consists of various commands, each performing a specific task.
You can get help on the command line by adding `--help` to any ibis related command.
The help message for `ibis` as a whole is:
```
compressible computational fluid dynamics
Usage: ibis [OPTIONS] [SUBCOMMAND]

Options:
  -h,--help                   Print this help message and exit
  -v,--version                print the version

Subcommands:
  clean                       clean the simulation
  prep                        prepare the simulation
  run                         run the simulation
  post                        post-process the simulation
```

## prep
`ibis prep` executes `job.py`, and writes detailed configuration files based on the settings in the python script.

```
prepare the simulation
Usage: ibis prep [OPTIONS]

Options:
  -h,--help                   Print this help message and exit
```

## run
`ibis run` reads the detailed configuration files written by the preparation stage, and executes the simulation

```
run the simulation
Usage: ibis run [OPTIONS]

Options:
  -h,--help                   Print this help message and exit
```

## post
`ibis post` performs post-processing of the simulation.
There are various sub-commands available for various types of post-processing (discussed below).

```
post-process the simulation
Usage: ibis post [OPTIONS] SUBCOMMAND

Options:
  -h,--help                   Print this help message and exit

Subcommands:
  plot                        write simulatioin files to visualisation format
```

### plot
`ibis post plot` writes the flow solutions written during the simulation to a visualisation format.

```
write simulatioin files to visualisation format
Usage: ibis post plot [OPTIONS]

Options:
  -h,--help                   Print this help message and exit
  -f,--format format (default: vtk-binary)
                              File format
  --add str ...               Extra variables to add to plot
```

The available file formats are:
  + `vtk-binary` (default)
  + `vtk-text`

By default, the pressure, temperature, speed of sound, Mach number, velocity, and energy are plotted.
Additional variables which may be added (via `--add`) are:
  + `viscous_grad_vx`
  + `viscous_grad_vy`
  + `viscous_grad_vz`
  + `viscous_grad_v` (add the gradient of all 3 velocity components)
  + `convective_grad_vx`
  + `convective_grad_vy`
  + `convective_grad_vz`
  + `convective_grad_v` (add the gradient of all 3 velocity components)
  + `cell_centre`

### plot_residuals
`ibis post plot_residuals` creates a plot of the residuals for the current simulation.
This requires numpy and matplotlib be installed.
Note that the active version of python when the code is compiled is used, so installing numpy and matplotlib in a virtual environment may not work.
```
plot simulation residuals
Usage: ibis post plot_residuals [OPTIONS]

Options:
  -h,--help                   Print this help message and exit
```

## clean
`ibis clean` cleans out the automatically generated files.
```
clean the simulation
Usage: ibis clean [OPTIONS]

Options:
  -h,--help                   Print this help message and exit
```
