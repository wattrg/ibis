# Ibis Reference Manual

## Directory Structure
`ibis` runs one simulation per directory.
`ibis` looks for specific files in that directory, and will automatically generate files.

The main configuration file `ibis` looks for is `job.py`.
This file contains the settings for the simulation.
```
|-- job_directory
    |-- job.py
```
