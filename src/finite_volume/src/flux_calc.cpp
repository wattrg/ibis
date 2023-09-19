#include <doctest/doctest.h>
#include "flux_calc.h"

template <typename T>
void compute_flux(FlowStates<T> &left, FlowStates<T> &right, 
                  ConservedQuantities<T> &flux, 
                  Interfaces<T> &faces, FluxCalculator flux_calc, bool three_d)
{
    transform_to_local_frame(left.vel, faces.norm(), faces.tan1(), faces.tan2());
    transform_to_local_frame(right.vel, faces.norm(), faces.tan1(), faces.tan2());

    switch (flux_calc) {
        case FluxCalculator::Hanel:
            hanel(left, right, flux, three_d);
            break;
    }

    // do we need to transform the flow states back to the global reference frame?
}

TEST_CASE("hanel") {
    CHECK(1 == 1);
}
