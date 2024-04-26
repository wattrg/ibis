---
title: Running a simulation
permalink: /docs/getting_started/run
subtitle: Getting Started
menubar: getting_started_menu
toc: true
---

Running an `ibis` simulation is done in three parts:
  1. Pre-processing
  2. Execution
  3. Post-processing

## Pre-processing
To run your first simulation, change directory into one of the examples.
```
cd examples/wedge
```
The first thing is to prepare the simulation.
For this, we need the grid.
`ibis` has no grid generation capabilities, so you need to generate the grid using some other software.
For this example, `grid.py` uses `gmsh` to generate the grid.
First install `gmsh` through `pip`, then execute the script:
```
python grid.py
```
This will write a `su2` file.

Now you can prepare the `ibis` job.
This is done with the following command:
```
ibis prep
```
This will look for `job.py`, execute it, and write detailed configuration, grid, and flow files.

## Execution
Now it is time to run the simulation.
This is done with the following command:
```
ibis run
```
This command looks for the files generated in the pre-processing stage, and runs the simulation.
It will periodically write flow solutions out (as often as you asked it to).

> *NOTE*:
> This stage doesn't look at `job.py`.
> If you make changes to that, you have to re-run the pre-processing stage.

## Post-processing
The flow solutions written during the simulation are in native `ibis` format, designed to write as little information as possible for efficiency.
Post-processing takes this information and recreates the full flow state, and writes more interesting information in a more widely used format. 
Currently, only `vtk` format is implemented.

To generate `vtk` files, run:
```
ibis post plot
```

This will generate `vtk` files in a directory called `plot`.
You can open `plot/plot.pvd` using `paraview` to see the results of the simulation.
