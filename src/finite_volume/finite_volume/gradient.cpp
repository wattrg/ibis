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
}
