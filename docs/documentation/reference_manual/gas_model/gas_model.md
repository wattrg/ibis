---
menubar: reference_manual_menu
---
# Gas Model

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

Only two of the fields need to be filled in by the user; the remainder can be computed using a `IdealGas` ([see here](/documentation/user_guide/gas_model/gas_model#IdealGas))

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
>   + `vy`: `float` the velocity in the y direction in (m/s). Defaults to \
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

The `IdealGas` can perform calculations on a `GasState` object:

### ```GasModel.update_from_pT(gas_state)```

Update a gas state from its pressure and temperature

> Positional arguments:\
> gas_state: `GasState` object. The pressure and temperature should already be set, and the density and energy will be updated.
> 
> Returns: None
