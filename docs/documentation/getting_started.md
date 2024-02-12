# Getting Started
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

### Configure
`ibis` uses `cmake` to configure things like the architecture you will run the code on.
Regardless of the architecture you will be using, `ibis` does not support in-tree builds.
So, regardless of your architecture, begin with:
```
mkdir build
cd build
```

Now you have to decide what architecture you want. 
This will most likely be whatever brand of GPU you have, but you may chose to use the CPU flavour for debugging.
The available architectures are:
  + `CUDA` (NVIDIA GPUs)
  + `HIP` (AMD GPUs)
  + `OPENMP` (Parallel on the CPU)
  + `SERIAL` (Serial on the CPU)

The next step is to configure the code.
Make sure to follow the configuration steps for your chosen architecture.

#### Configure CUDA, CPU
With your chosen architecture, run
```
cmake .. -DKokkos_ENABLE[ARCH]=ON
```

#### Configure HIP
If using an AMD GPU, make sure that `ROCM_PATH` is set to `/opt/rocm` before proceeding.
If you are using any machine but Bunya, configure `ibis` with:
```
cmake .. -DKokkos_ENABLE_HIP=ON -DCMAKE_CXX_COMPILER=hipcc
```
If you are compiling for the AMD GPUs on Bunya, the filesystem module has to be linked manually.
Configure `ibis` with:
```
cmake .. -DKokkos_ENABLE_HIP=ON -DCMAKE_CXX_COMPILER=hipcc -DIbis_LINK_FS=ON
```

### Compile
Finally, to compile and install (regardless of architecture):
```
make -j install
```

## Running simulations
I'll add an example of running a simulation here soon...
