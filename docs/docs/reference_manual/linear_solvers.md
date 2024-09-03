---
toc: true
title: Linear Solvers
---

## Gmres
Un-preconditioned GMRES.

```
Gmres(tol=1e-14, max_iters=50)
```

The arguments are described below

### tol
The tolerance for convergence of the linear system

> Type: `float`\
> Default: 1e-2

### max_iters
The maximum number of iterations to solve the linear system

> Type: `int`\
> Default: 50  

## FGmres
Flexible GMRES.

```
FGmres(
  tolerance=1e-14,
  max_iters=50,
  max_preconditioner_iters=5,
  preconditioner_tolerance=1e-1
)
```

The arguments are described below

### tolerance
The tolerance for convergence of the linear system

> Type: `float`\
> Default: 1e-2

### max_iters
The maximum number of iterations to solve the linear system

> Type: `int`\
> Default: 50  

### max_preconditioner_iters
The maximum number of iterations for preconditioning

> Type: `int`\
> Default: 5

### preconditioner_tolerance
The tolerance for converging the preconditioner system

> Type: `float`\
> Default: 1e-1
