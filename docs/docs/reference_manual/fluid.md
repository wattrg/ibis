---
title: Fluid settings
toc: true
---

# Convective Flux
The convective flux is configured by setting `config.convective_flux` to an instance of the `ConvectiveFlux` class in `job.py`.
For example:

```
config.convective_flux = ConvectiveFlux(
    flux_calculator = Ausmdv(),
    reconstruction_order = 2,
    limiter = BarthJespersen(),
    thermo_interpolator=ThermoInterp.RhoU
)
```
If `config.convective_flux` is not set, all the default options will be used.
If the `ConvectiveFlux` object is only passed some of the options, the default options for the remaining settings will be used.

A description of the settings is given below.

### flux_calculator
The flux calculator the simulation will use.

> Type: `FluxCalculator`\
> Default: `Hanel()`\
> Options:
>  + `Ausmdv()`: This is a good low-dissipation flux calculator
>  + `Hanel()`: This is a fairly dissipative flux calculator, good for debugging troublesome simulations

### reconstruction_order
The slope reconstruction order for higher order accuracy.

> Type: `int`\
> Default: `2`\
> Options: `1` or `2`

### limiter
The slope limiter the simulation will use, if `reconstruction_order = 2`

> Type: `Limiter`\
> Default: `BarthJespersen()`\
> Options:
>  + `BarthJespersen(epsilon=1e-25)`: A very strict slope limiter
>    + `epsilon` is a small number to avoid division by zero, and can control the amount of limiting done. The default is 1e-25.
>  + `Unlimited()`: Disable slope limiting

### thermo_interpolator
The two thermodynamic variables to interpolate when performing reconstruction.
The other thermodynamic variables are computed using the gas model

> type: `ThermoInterp` | `str`\
> Default: `ThermoInterp.RhoU` | `"rho_u"`\
> Options:
>  + `ThermoInterp.RhoU` / `"rho_u"`
>  + `ThermoInterp.RhoT` / `"rho_T"`
>  + `ThermoInterp.RhoP` / `"rho_p"`
>  + `ThermoInterp.pT` / `"p_T"`

## Viscous Flux
The viscous flux is configured by setting `config.viscous_flux` to an instance of the `ViscousFlux` class in `job.py`.
For example:
```
config.viscous_flux = ViscousFlux(
    enabled = true
)
```
If `config.viscous_flux` is not set, all the default options will be used.
If the `ViscousFlux` object is only passed some of the options, the default options for the remaining settings will be used.
A description of the settings is given below.

A description of the settings is given below.

### enabled
Whether to enable viscous effects or not

> Type: `bool`\
> Default: `False`

