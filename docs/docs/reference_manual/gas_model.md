---
title: Gas Models
toc: true
---
## GasState
A `GasState` object describes the thermodynamic state of some gas.
For example:
```
gas_state = GasState()
gas_state.p = 101325
gas_state.T = 300
```

A `GasState` contains the following information:
> `p` (Pa): pressure\
> `T` (K): temperature\
> `rho` (kg/m^3): density\
> `energy` (J/kg): internal energy

Only two of the fields need to be filled in by the user; the remainder can be computed using a `IdealGas` ([see here](/docs/reference_manual/gas_model#IdealGas))

## FlowState
A `FlowState` describes the thermodynamic state of some gas, and the gas's velocity.
A `FlowState` can be constructed via:
> `FlowState(gas, vx=..., vy=... vz=...)`
>
> Positional arguments:
>   + `gas`: `GasState` object with the thermodynamic state
>
> Optional arguments:
>   + `vx`: `float` the velocity in the x direction in (m/s). Defaults to 0
>   + `vy`: `float` the velocity in the y direction in (m/s). Defaults to 0
>   + `vz`: `float` the velocity in the z direction in (m/s). Defaults to 0


## IdealGas
A `IdealGas` describes the ideal gas properties of the gas.
An `IdealGas` object can be created in one of two ways; from the gas constant, or from a species name.

Gas constant:
> `IdealGas(R = ...)`
>
> Keyword arguments:
>   + `R` (float): the gas constant (J/kg)

Species name:
> `IdealGas(species = ...)`
>
> Keyword arguments:
>   + `species` (`String`): the name of the species

The `IdealGas` can perform calculations on a `GasState` object.
For example:
```
gas_model = IdealGas(species='air')
gas_state = GasState()
gas_state.p = 101325
gas_state.T = 300
gas_model.update_thermo_frompT(gas_state)
```

The available methods on a gas model are:

### ```update_from_pT(gas_state)```

Update a gas state from its pressure and temperature

> Positional arguments:\
> gas_state: `GasState` object. The pressure and temperature should already be set, and the density and energy will be updated.
> 
> Returns: None

### ```update_from_rhoT(gas_state)```

Update a gas state from its density and temperature

> Positional arguments:\
> gas_state: `GasState` object. The density and temperature should already be set, and the pressure and energy will be updated
>
> Returns: None

### ```update_from_rhop(gas_state)```

  Update a gas state from its density and pressure

  > Positional arguments:\
  > gas_state: `GasState` object. The density and pressure should already be set, and the temperature and energy will be calculated
  >
  > Returns: None
