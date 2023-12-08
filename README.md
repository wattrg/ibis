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
  1. A c++ compiler (with c++17 support; g++ 11.4 works)
  2. cmake (at least version 3.20.1)
  3. make (or any cmake compatible generator you like)
  4. Python
  5. Native compilers for the target architecture (e.g. cuda, hip)

### Download
To download the code:
```
git clone git@github.com:wattrg/ibis.git --recurse-submodules
```
If you missed the `--recurse-submodules`, you can initialise the submodules them with

```
git submodule update --init --recursive
```

### build
To compile `ibis`, from the root of the repository, run:

```
mkdir build
cd build
cmake .. -DKokkos_ENABLE_[ARCH]=ON
make install
```
where `[ARCH]` refers to the architecture to build for. For example, `[ARCH]` could be `SERIAL`, `OPENMP`, `CUDA`. 

If compiling with `HIP`, you'll need to make sure the `ROCM_PATH` environment variable is set to `/opt/rocm`. 
And you'll also have to change the `cmake` line to:

```
cmake .. -DKokkos_ENABLE_HIP=ON -DCMAKE_CXX_COMPILER=hipcc
```

This will compile and install ibis. You probably want to add the install directory (by default this is in the `inst` folder in the root of the repository) to your `PATH`.

## Gridding
To build grids, you can use what ever software you wish, as long as it can generate `su2` files.
The examples use the gmsh python api, which can be installed with

```
pip install gmsh
```
