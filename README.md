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

To build grids, you can use what every software you wish, as long as it can generate su2 files.
The examples use the gmsh python api, which can be installed with

```
pip install gmsh
```

## build
`ibis` uses `cmake`. To compile `ibis` on linux:

```
mkdir build
cd build
cmake .. -DKokkos_ENABLE_ARCH=ON
make install
```

where `ARCH` refers to the architecture to build for. For example, `ARCH` could be `OPENMP`, `CUDA`, or `HIP`.

This will compile and install ibis. You probably want to add the install directory (by default this is in the `inst` folder in the root of the repository) to your `PATH`.
