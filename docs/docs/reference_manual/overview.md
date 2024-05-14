---
title: Overview
toc: true
---

## Directory Structure
`ibis` runs one simulation per directory.
`ibis` looks for specific files in that directory, and will automatically generate files.

The main configuration file `ibis` looks for is `job.py`.
This file contains the settings for the simulation.
```
|-- job_directory
    |-- job.py
```

After preparing a simulation, a number of files will be generated, and the directory should look something like:
```
|-- job_directory
    |-- job.py
    |-- config/
    |-- grid/
    |-- flow/
```

When `ibis` begins a simulation, it no longer looks at `job.py`, only the generated config files.
So if you wish to change a setting, make sure to run `ibis prep` again.

## Typical Workflow
  1. Build the grid. Any grid generation software that can export su2 files will work. Currently, the grid must be a single block. The dimensionality of the grid sets the dimensionality of the simulation
  2. Prepare the simulation with `ibis prep`
  3. Run the simulation with `ibis run`
  4. Post-process the simulation with `ibis post plot`
