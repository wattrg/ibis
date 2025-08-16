#include <finite_volume/finite_volume.h>
#include <gas/gas_model.h>
#include <gas/transport_properties.h>
#include <simulation/simulation.h>
#include <util/numeric_types.h>

template <typename T, class MemModel>
Sim<T, MemModel>::Sim(GridBlock<MemModel, T> grid_, json config) : grid(grid_) {
    fv = FiniteVolume<T, MemModel>(grid, config);
    gas_model = IdealGas<T>(config.at("gas_model"));
    trans_prop = TransportProperties<T>(config.at("transport_properties"));
}

template struct Sim<Ibis::real, SharedMem>;
template struct Sim<Ibis::real, Mpi>;
template struct Sim<Ibis::dual, SharedMem>;
template struct Sim<Ibis::dual, Mpi>;
