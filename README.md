# aeolus
A soon-to-be CFD solver.

## Installation
### Dependancies
To build `aeolus`, you will need to have:
  1. A c++ compiler
  2. cmake
  3. Kokkos

#### Install Kokkos
To install Kokkos:

```
git clone git@github.com:kokkos/kokkos.git
mkdir kokkos_inst
cd kokkos
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../../kokkos_inst
make install
```

## build
`aeolus` uses `cmake`. To compile `aeolus` on linux:

```
mkdir build
cd build
cmake .. -DKokkos_ROOT=../kokkos_inst
make
```

This will compile the executable. You can run it with `./aeolus`
