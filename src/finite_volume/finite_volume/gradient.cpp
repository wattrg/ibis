#include <doctest/doctest.h>
#include <finite_volume/gradient.h>
#include <grid/grid.h>

#include <Kokkos_Core.hpp>
#include <nlohmann/json.hpp>

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
    auto block_host = block_dev.host_mirror();
    block_host.deep_copy(block_dev);
    WLSGradient<double> wls_gradient(block_dev);
    Kokkos::View<double*> values("values", 21);
    auto values_host = Kokkos::create_mirror_view(values);
    values_host(0) = 1.0;
    values_host(1) = 2.0;
    values_host(2) = 3.0;
    values_host(3) = 1.5;
    values_host(4) = 2.5;
    values_host(5) = 3.5;
    values_host(6) = 2.0;
    values_host(7) = 3.0;
    values_host(8) = 4.0;
    auto bottom = block_host.ghost_cells("slip_wall_bottom");
    values_host(bottom(0)) = 0.5;
    values_host(bottom(1)) = 1.5;
    values_host(bottom(2)) = 2.5;
    auto top = block_host.ghost_cells("slip_wall_top");
    values_host(top(0)) = 2.5;
    values_host(top(1)) = 3.5;
    values_host(top(2)) = 4.5;
    auto inflow = block_host.ghost_cells("inflow");
    values_host(inflow(0)) = 0.0;
    values_host(inflow(1)) = 0.5;
    values_host(inflow(2)) = 1.0;
    auto outflow = block_host.ghost_cells("outflow");
    values_host(outflow(0)) = 4.0;
    values_host(outflow(1)) = 4.5;
    values_host(outflow(2)) = 5.0;

    Kokkos::deep_copy(values, values_host);

    Kokkos::View<double*> grad_x("grad_x", 9);
    Kokkos::View<double*> grad_y("grad_y", 9);
    Kokkos::View<double*> grad_z("grad_z", 9);
    wls_gradient.compute_gradients(block_dev, values, grad_x, grad_y, grad_z);
    auto grad_x_host = Kokkos::create_mirror_view(grad_x);
    auto grad_y_host = Kokkos::create_mirror_view(grad_y);
    Kokkos::deep_copy(grad_x_host, grad_x);
    Kokkos::deep_copy(grad_y_host, grad_y);
    for (int i = 0; i < 9; i++) {
        CHECK(grad_x_host(i) == doctest::Approx(1.0));
        CHECK(grad_y_host(i) == doctest::Approx(0.5));
    }
}
