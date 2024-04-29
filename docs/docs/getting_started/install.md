---
title: Install
subtitle: Getting Started
toc: true
---

## Dependancies
To build `ibis`, you will need to have:
  1. A c++ compiler (with c++17 support; g++ 11.4 works)
  2. cmake (at least version 3.20.1)
  3. make (or any cmake compatible generator you like)
  4. Python
  5. Native compilers for the target architecture (e.g. cuda, hip)

## Download
To download the code:
```
git clone git@github.com:wattrg/ibis.git --recurse-submodules
```
If you missed the `--recurse-submodules`, you can initialise the submodules them with

```
git submodule update --init --recursive
```

## Configure
Ibis does not allow you to build the code directly in the source code tree.
We need to make a separate directory to build the code in.
In the root of the repository, make a new directory:
```
mkdir build
cd build
```

`ibis` uses `cmake` to configure things like the architecture you will run the code on.
The available architectures are:
  + `CUDA` (NVIDIA GPUs)
  + `HIP` (AMD GPUs)
  + `OPENMP` (Parallel on the CPU)
  + `SERIAL` (Serial on the CPU)

The next step is to configure the code.
Make sure to follow the configuration steps for your chosen architecture.

### Configure for CUDA or CPU
With your chosen architecture, run
```
cmake .. -DKokkos_ENABLE_[ARCH]=ON
```

### Configure for HIP
If using an AMD GPU, make sure that `ROCM_PATH` is set to `/opt/rocm` before proceeding.
If you are using any machine but Bunya, configure `ibis` with:
```
cmake .. -DKokkos_ENABLE_HIP=ON -DCMAKE_CXX_COMPILER=hipcc
```

### Configure for HIP on Bunya
If you are compiling for the AMD GPUs on Bunya, the filesystem module has to be linked manually.
Configure `ibis` with:
```
cmake .. -DKokkos_ENABLE_HIP=ON -DCMAKE_CXX_COMPILER=hipcc -DIbis_LINK_FS=ON
```

## Compile and Install
Finally, to compile and install (regardless of architecture):
```
make -j install
```

Finally, add the install location to your system path.
By default, the install location is `<path_to_ibis_repo>/inst`
