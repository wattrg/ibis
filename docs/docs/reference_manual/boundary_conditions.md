---
title: Boundary conditions
toc: true
---
## Boundary Conditions
Boundary conditions are set in the `config.grid` in `job.py`. 
See the [grid config](grid) for details on attaching boundary conditions to the grid.
This page lists the available boundary conditions.

### slip_wall
A slip wall boundary condition.
Constructed by:
```
slip_wall()
```
### adiabatic_no_slip_wall
A no-slip wall, with no heat-transfer at the wall. Constructed by:
```
adiabatic_no_slip_wall()
```

### fixed_temperature_no_slip_wall
A no-slip wall, with a fixed wall temperature. Constructed by:
```
fixed_temperature_no_slip_wall(temperature)
```

Arguments:\
`temperature`: (double) the desired wall temperature

### supersonic_inflow
A supersonic inflow boundary condition.
This boundary condition assumes no information travels out of the domain.
Constructed by:
```
supersonic_inflow(flow_state)
```

Arguments:\
`inflow`: [FlowState](gas_model#flowstate) object with the desired supersonic flow state

### supersonic_outflow
A supersonic outflow boundary condition. 
This boundary condition assumes no information travels into the domain.
Constructed by:
```
supersonic_outflow()
```

### boundary_layer_inflow
An inflow boundary condition with a specified 1D profile, meant to start the domain part way along a boundary layer
```
boundary_layer_inflow(pressure, height, temperature_profile, velocity_profile)
```

Arguments:\
`pressure`: (double) The pressure of the flow (constant across the boundary layer)\
`height`: (list[double]) The heights the temperature and velocity are sampled at in metres\
`temperature_profile`: (list[double]) The temperatures (in Kelvin) of the flow at distances from the wall corresponding to the `heights`\
`velocity_profile`: (list[double]) The x component of the velocity (in m/s) of the flow at distance from the wall corresponding to the `heights`

The distance from the wall is computed as just the y component of the cell centre in the global reference frame, so the wall must lie on the x-z plane.
