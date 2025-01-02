#ifndef IBIS_PARALLEL_H
#define IBIS_PARALLEL_H

#include <parallel/parallel_fwd.h>
#include <parallel/reductions.h>

namespace Ibis {

template <typename Scalar, class MemoryModel = Ibis::DefaultMemModel>
using DistributedMin = Distributed::DistributedReduction<Min<Scalar>, MemoryModel>;

template <typename Scalar, class MemoryModel = Ibis::DefaultMemModel>
using DistributedMax = Distributed::DistributedReduction<Max<Scalar>, MemoryModel>;

template <typename Scalar, class MemoryModel = Ibis::DefaultMemModel>
using DistributedSum = Distributed::DistributedReduction<Sum<Scalar>, MemoryModel>;

}  // namespace Ibis

#endif
