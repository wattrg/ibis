#include "cubic_spline.h"
#include <stdexcept>
#include "Kokkos_Core_fwd.hpp"
#include <doctest/doctest.h>

CubicSpline::CubicSpline(std::vector<double> x_vec, std::vector<double> y_vec) {
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

    Kokkos::View<double*, Kokkos::DefaultHostExecutionSpace> x("x", n_pts);
    Kokkos::View<double*, Kokkos::DefaultHostExecutionSpace> y("y", n_pts);
    for (size_t i = 0; i < n_pts; i++) {
        x(i) = x_vec[i];
        y(i) = y_vec[i];
    }

    // we solve for the second derivative at internal points, so we
    // have n_points - 2 equations to solve
    size_t n_eqns = n_pts - 2;

    Kokkos::View<double*, Kokkos::DefaultHostExecutionSpace> diag("diag", n_eqns);
    Kokkos::View<double*, Kokkos::DefaultHostExecutionSpace> upper("upper", n_eqns - 1);
    Kokkos::View<double*, Kokkos::DefaultHostExecutionSpace> rhs("rhs", n_eqns);
    Kokkos::View<double*, Kokkos::DefaultHostExecutionSpace> y_dash_dash("y_dash_dash",
                                                                         n_eqns);
    x_ = Kokkos::View<double*>("CubicSpline::x_", n_pts);
    y_ = Kokkos::View<double*>("CubicSpline::y_", n_pts);
    y_dash_dash_ = Kokkos::View<double*>("CubicSpline::y''", n_eqns);


    // fill in the arrays to set up the system of equations
    double delta_xi, delta_yi;
    for (size_t i = 0; i <= n_eqns; i++) {
        delta_xi = x(i + 1) - x(i);
        delta_yi = y(i + 1) - y(i);

        if (i != n_eqns) {
            diag(i) += delta_xi;
            rhs(i) -= delta_yi / delta_xi;
        }

        if (i == 0) continue;

        rhs(i - 1) += delta_yi / delta_xi;
        diag(i - 1) += delta_xi;
        upper(i - 1) += delta_xi;
    }
    for (size_t i = 0; i < n_eqns; i++) {
        rhs(i) *= 6;
        diag(i) *= 2;
    }

    // use the Thomas alorithm to solve the tri-diagonal system of equations
    // sweep forward, eliminating bottom diagonal
    double w;
    for (size_t i = 1; i < n_eqns; i++) {
        w = upper(i - 1) / diag(i - 1);
        diag(i) = diag(i) - w * upper(i - 1);
        rhs(i) = rhs(i) - w * rhs(i - 1);
    }

    // back substitution
    y_dash_dash(n_pts - 2) = rhs(n_eqns - 1) / diag(n_eqns - 1);
    for (int i = n_eqns - 2; i >= 0; i--) {
        y_dash_dash(i + 1) = (rhs(i) - upper(i) * y_dash_dash(i + 2)) / diag(i);
    }

    // copy data to the device
    Kokkos::deep_copy(x_, x);
    Kokkos::deep_copy(y_, y);
    Kokkos::deep_copy(y_dash_dash_, y_dash_dash);
}


TEST_CASE("CubicSpline"){
    std::vector<double> x {1.0, 2.0, 3.0, 4.0};
    std::vector<double> y {4.0, 6.0 ,8.0, 10.0};

    CubicSpline spline(x, y);

    Kokkos::View<double*> results_dev ("Results", 6);
    Kokkos::View<double*, Kokkos::DefaultHostExecutionSpace> results_host("Results_host", 6);

    Kokkos::parallel_for("Run cubic spline", 1, KOKKOS_LAMBDA(const size_t i) {
        (void) i;
        results_dev(0) = spline.eval(1.0);
        results_dev(1) = spline.eval(1.5);                         
        results_dev(2) = spline.eval(2.0);
        results_dev(3) = spline.eval(4.0);
        results_dev(4) = spline.eval(5.0);
        results_dev(5) = spline.eval(0.0);
    });

    Kokkos::deep_copy(results_host, results_dev);

    CHECK(results_host(0) == doctest::Approx(4.0));
    CHECK(results_host(1) == doctest::Approx(5.0));
    CHECK(results_host(2) == doctest::Approx(6.0));
    CHECK(results_host(3) == doctest::Approx(10.0));
    CHECK(results_host(4) == doctest::Approx(10.0));
    CHECK(results_host(5) == doctest::Approx(4.0));
}
