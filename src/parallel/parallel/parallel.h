#ifndef IBIS_PARALLEL_H
#define IBIS_PARALLEL_H

#include <parallel/parallel_fwd.h>
#include <parallel/reductions.h>
#include <parallel/shared_memory.h>
#include <parallel/distributed_memory.h>

namespace Ibis {

void initialise(int argc, char** argv);

void finalise();

template <typename Scalar, class MemoryModel = Ibis::DefaultMemModel>
using DistributedMin = Distributed::DistributedReduction<Min<Scalar>, MemoryModel>;

template <typename Scalar, class MemoryModel = Ibis::DefaultMemModel>
using DistributedMax = Distributed::DistributedReduction<Max<Scalar>, MemoryModel>;

template <typename Scalar, class MemoryModel = Ibis::DefaultMemModel>
using DistributedSum = Distributed::DistributedReduction<Sum<Scalar>, MemoryModel>;

}  // namespace Ibis

#endif
