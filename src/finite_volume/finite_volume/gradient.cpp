#include <Kokkos_Core.hpp>
#include <doctest/doctest.h>
#include <nlohmann/json.hpp>
#include <finite_volume/gradient.h>
#include <grid/grid.h>

json build_config() {
    json config{};
    json boundaries{};
    json slip_wall{};
    json inflow{};
    json outflow{};
    slip_wall["ghost_cells"] = true;
    inflow["ghost_cells"] = true;
    outflow["ghost_cells"] = true;
    boundaries["slip_wall_bottom"] = slip_wall;
    boundaries["slip_wall_top"] = slip_wall;
    boundaries["inflow"] = inflow;
    boundaries["outflow"] = outflow;
    config["boundaries"] = boundaries;
    return config;
}

TEST_CASE("gradient") {
    json config = build_config();
    GridBlock<double> block_dev("../../../src/grid/test/grid.su2", config);
    WLSGradient<double> wls_gradient(block_dev);
    Kokkos::View<double*> values ("values", 21);
    Kokkos::View<double*> grad_x ("grad_x", 9);
    Kokkos::View<double*> grad_y ("grad_y", 9);
    Kokkos::View<double*> grad_z ("grad_z", 9);
    wls_gradient.compute_gradients(block_dev, values, grad_x, grad_y, grad_z);
    auto grad_x_host = Kokkos::create_mirror_view(grad_x);
    CHECK(grad_x_host(4) == 0.0);
}
