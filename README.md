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
  1. A c++ compiler (with c++17 support, the default one on Ubuntu is good)
  2. cmake (at least version 3.20.1)
  3. make (or any cmake compatible generator you like)

### build
To compile `ibis`, from the root of the repository, run:

```
mkdir build
cd build
cmake .. -DKokkos_ENABLE_[ARCH]=ON
make install
```
where `[ARCH]` refers to the architecture to build for. For example, `[ARCH]` could be `SERIAL`, `OPENMP`, `CUDA`, or `HIP`.

This will compile and install ibis. You probably want to add the install directory (by default this is in the `inst` folder in the root of the repository) to your `PATH`.

## Gridding
To build grids, you can use what every software you wish, as long as it can generate `su2` files.
The examples use the gmsh python api, which can be installed with

```
pip install gmsh
```
