#include <doctest/doctest.h>
#include <util/cubic_spline.h>

#include <stdexcept>

// #include "Kokkos_Core_fwd.hpp"

CubicSpline::CubicSpline(std::vector<Ibis::real> x_vec, std::vector<Ibis::real> y_vec) {
    // allocate memory on the device, and mirrors on the host for setting
    // up the interpolation
    size_t n_pts = x_vec.size();
    n_pts_ = n_pts;
    if (n_pts < 4) {
        throw new std::runtime_error("Insufficient points for cubic spline");
    }

    if (y_vec.size() != n_pts) {
        throw new std::runtime_error(
            "Cubic spline has different number of x and y points");
    }

    x_min_ = x_vec[0];
    x_max_ = x_vec[n_pts - 1];
    y_min_ = y_vec[0];
    y_max_ = y_vec[n_pts - 1];

    x_ = Kokkos::View<Ibis::real*>("CubicSpline::x_", n_pts);
    y_ = Kokkos::View<Ibis::real*>("CubicSpline::y_", n_pts);

    auto x = Kokkos::create_mirror_view(x_);
    auto y = Kokkos::create_mirror_view(y_);
    for (size_t i = 0; i < n_pts; i++) {
        x(i) = x_vec[i];
        y(i) = y_vec[i];
    }

    Kokkos::View<Ibis::real*, Kokkos::DefaultHostExecutionSpace> u("u", n_pts);
    y_dash_dash_ = Kokkos::View<Ibis::real*>("CubicSpline::y''", n_pts);
    auto y_dash_dash = Kokkos::create_mirror_view(y_dash_dash_);

    // Formation (and decompoistion) of the tri-diagonal system.
    // This implementation is from the cubic spline interpolation section of the 3rd
    // edition of Numerical Methods in C (p. 123), using natural boundary conditions.
    for (size_t i = 1; i < n_pts - 1; i++) {
        Ibis::real sigma = (x(i) - x(i - 1)) / (x(i + 1) - x(i - 1));
        Ibis::real p = sigma * y_dash_dash(i - 1) + 2.0;
        y_dash_dash(i) = (sigma - 1) / p;
        u(i) =
            (y(i + 1) - y(i)) / (x(i + 1) - x(i)) - (y(i) - y(i - 1)) / (x(i) - x(i - 1));
        u(i) = (6.0 * u(i) / (x(i + 1) - x(i - 1)) - sigma * u(i - 1)) / p;
    }

    // backsubstitution
    y_dash_dash(n_pts - 1) = 0.0;
    for (int k = n_pts - 2; k >= 0; k--) {
        y_dash_dash(k) = y_dash_dash(k) * y_dash_dash(k + 1) + u(k);
    }

    // copy data to the device
    Kokkos::deep_copy(x_, x);
    Kokkos::deep_copy(y_, y);
    Kokkos::deep_copy(y_dash_dash_, y_dash_dash);
}

TEST_CASE("CubicSpline") {
    std::vector<Ibis::real> x{1.0, 2.0, 3.0, 4.0, 8.0};
    std::vector<Ibis::real> y{4.0, 6.0, 8.0, 10.0, 18.0};

    CubicSpline spline(x, y);

    Kokkos::View<Ibis::real*> results_dev("Results", 7);
    Kokkos::View<Ibis::real*, Kokkos::DefaultHostExecutionSpace> results_host(
        "Results_host", 7);

    Kokkos::parallel_for(
        "Run cubic spline", 1, KOKKOS_LAMBDA(const size_t i) {
            (void)i;
            results_dev(0) = spline.eval(1.0);
            results_dev(1) = spline.eval(1.5);
            results_dev(2) = spline.eval(2.0);
            results_dev(3) = spline.eval(4.0);
            results_dev(4) = spline.eval(5.0);
            results_dev(5) = spline.eval(0.0);
            results_dev(6) = spline.eval(7.0);
        });

    Kokkos::deep_copy(results_host, results_dev);

    CHECK(results_host(0) == doctest::Approx(4.0));
    CHECK(results_host(1) == doctest::Approx(5.0));
    CHECK(results_host(2) == doctest::Approx(6.0));
    CHECK(results_host(3) == doctest::Approx(10.0));
    CHECK(results_host(4) == doctest::Approx(12.0));
    CHECK(results_host(5) == doctest::Approx(4.0));
    CHECK(results_host(6) == doctest::Approx(16.0));
}

TEST_CASE("CubicSpline2") {
    std::vector<Ibis::real> x{
        0.000000000, 0.000286575, 0.000642487, 0.001041700, 0.001466750, 0.001900160,
        0.002326750, 0.003120560, 0.003816400, 0.004433000, 0.005547930, 0.006617530,
    };
    std::vector<Ibis::real> y{
        0.00000, 153.017, 319.372, 488.652, 654.953, 811.480,
        951.827, 1167.76, 1295.30, 1356.55, 1387.81, 1390.00,
    };

    CubicSpline spline(x, y);

    Kokkos::View<Ibis::real*> results_dev("Results", 6);
    Kokkos::View<Ibis::real*, Kokkos::DefaultHostExecutionSpace> results_host(
        "Results_host", 6);

    Kokkos::parallel_for(
        "Run cubic spline", 1, KOKKOS_LAMBDA(const size_t i) {
            (void)i;
            results_dev(0) = spline.eval(0.0);
            results_dev(1) = spline.eval(0.000286575);
            results_dev(2) = spline.eval(0.000642487);
            results_dev(3) = spline.eval(0.005547930);
            results_dev(4) = spline.eval(0.1);
            results_dev(5) = spline.eval(-0.1);
        });

    Kokkos::deep_copy(results_host, results_dev);

    CHECK(results_host(0) == doctest::Approx(0.0));
    CHECK(results_host(1) == doctest::Approx(153.017));
    CHECK(results_host(2) == doctest::Approx(319.372));
    CHECK(results_host(3) == doctest::Approx(1387.81));
    CHECK(results_host(4) == doctest::Approx(1390));
    CHECK(results_host(5) == doctest::Approx(0.0));
}
