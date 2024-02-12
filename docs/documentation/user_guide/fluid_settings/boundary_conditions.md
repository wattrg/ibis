---
title: Ibis
subtitle: User Guide
menubar: user_guide_menu
toc: true
---

# Boundary Conditions
Boundary conditions are set in the `config.grid` in `job.py`. 
See the [grid config](/documentation/user_guide/grid/grid) for details on attaching boundary conditions to the grid.
This page lists the available boundary conditions.

## slip_wall
A slip wall boundary condition.
Constructed by:
```
slip_wall()
```
## adiabatic_no_slip_wall
A no-slip wall, with no heat-transfer at the wall. Constructed by:
```
adiabatic_no_slip_wall()
```

## supersonic_inflow
A supersonic inflow boundary condition.
This boundary condition assumes no information travels out of the domain.
Constructed by:
```
supersonic_inflow(flow_state)
```

Arguments:\
`inflow`: [FlowState](/documentation/user_guide/gas_model/gas_model#FlowState) object with the desired supersonic flow state

## supersonic_outflow
A supersonic outflow boundary condition. 
This boundary condition assumes no information travels into the domain.
Constructed by:
```
supersonic_outflow()
```
