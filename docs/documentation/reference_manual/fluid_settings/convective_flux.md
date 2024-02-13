---
menubar: reference_manual_menu
---
# Convective Flux
The convective flux is configured by setting `config.convective_flux` to an instance of the `ConvectiveFlux` class in `job.py`.
For example:

```
config.convective_flux = ConvectiveFlux(
    flux_calculator = FluxCalculator.Ausmdv,
    reconstruction_order = 2,
    limiter = Limiter.BarthJespersen
)
```
If `config.convective_flux` is not set, all the default options will be used.
If the `ConvectiveFlux` object is only passed some of the options, the default options for the remaining settings will be used.

A description of the settings is given below.

## flux_calculator
The flux calculator the simulation will use.

> Type: `FluxCalculator` / `String`\
> Default: `FluxCalculator.Hanel` / `'hanel'`\
> Options:
>  + `FluxCalculator.Ausmdv` / `'ausmdv'`: This is a good low-dissipation flux calculator
>  + `FluxCalculator.Hanel` / `'hanel'`: This is a fairly dissipative flux calculator, good for debugging troublesome simulations

## reconstruction_order
The slope reconstruction order for higher order accuracy.

> Type: `int`\
> Default: `2`\
>  Options: `1` or `2`

## limiter
The slope limiter the simulation will use, if `reconstruction_order = 2`

>  Type: `Limiter` / `String`\
>  Default: `Limiter.BarthJespersen` / `'barth_jespersen'`\
> Options:
>  + `Limiter.BarthJespersen` / `'barth_jespersen'`: A very strict slope limiter
>  + `None` / `'none'`: Disable slope limiting
