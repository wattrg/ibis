#ifndef JFNK_H
#define JFNK_H

#include <finite_volume/conserved_quantities.h>
#include <gas/flow_state.h>
#include <gas/transport_properties.h>
#include <io/io.h>
#include <linear_algebra/linear_system.h>
#include <solvers/cfl.h>
#include <util/numeric_types.h>

#include <memory>

class Jfnk {
public:
    void step(LinearSystem& system);
    void solve(LinearSystem& system);

    size_t max_steps() const { return max_steps_; }

private:
    size_t max_steps_;
    std::unique_ptr<CflSchedule> cfl_;
};

#endif