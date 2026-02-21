#ifndef IBIS_PARALLEL_FWD_H
#define IBIS_PARALLEL_FWD_H

#include <parallel/memory_model_fwd.h>

#include <string>

namespace Ibis {

// The c++ standard doesn't allow partial specialisation of free-standing functions,
// so we can't partially specialise on the memory model. To get around this,
// we build these classes, which do allow partial specialisation. The free-standing
// functions will then use the class to delegate the work.
template <class ReducerType, class MemModel>
class Reducer {
    template <class FunctorType>
    inline auto execute(const std::string& str, const size_t work_count,
                        FunctorType functor) -> typename ReducerType::scalar_type;

    template <class PolicyType, class FunctorType>
    inline auto execute(const std::string& str, PolicyType& policy, FunctorType functor)
        -> typename ReducerType::scalar_type;
};

template <class ReducerType, class MemModel>
struct ReducerMap;

}  // namespace Ibis

#endif
