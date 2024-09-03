---
toc: true
title: Cfl Schedules
---

## ResidualBasedCfl
The CFL value grows as the residuals drop, so that as steady-state is approached, the CFL grows.

### growth_threshold
The residual drop required before beginning to grow the residual

> Type: `float`\
> Default: `0.1`

### power
The rate to grow the CFL at

> Type: `float`\
> Default: 0.8

### start_cfl
The CFL value to start

> Type: `float`\
> Default: 1.0

### max_cfl

> Type: `float`\
> Default: 1e8
