---
title: Ibis
subtitle: User Guide
menubar: user_guide_menu
---

# Viscous Flux
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

## enabled
Whether to enable viscous effects or not

> Type: `bool`\
> Default: `False`
