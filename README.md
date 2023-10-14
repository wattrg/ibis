# ibis
```
   MM      
  <' \___/| 
   \_  _/           _ _     _     
     ][            (_) |   (_)    
 ___/-\___          _| |__  _ ___ 
|---------|        | | '_ \| / __|
 | | | | |         | | |_) | \__ \
 | | | | |         |_|_.__/|_|___/
 | | | | |     
 | | | | |                
 |_______|
```

A compresible CFD solver

## Installation
### Dependancies
To build `ibis`, you will need to have:
  1. A c++ compiler
  2. cmake

## build
`ibis` uses `cmake`. To compile `ibis` on linux:

```
mkdir build
cd build
cmake .. -DKokkos_ENABLE_ARCH=ON
make
```

where `ARCH` refers to the architecture to build for. For example, `ARCH` could be `OPENMP`, `CUDA`, or `HIP`.

This will compile the executable. You can run it with `./ibis`
