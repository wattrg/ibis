#include <Kokkos_Core.hpp>
#include <cassert>
#include <vector>

class CubicSpline {
public:
    CubicSpline() {}
    CubicSpline(std::vector<double> x, std::vector<double> y);

    KOKKOS_FUNCTION
    double eval(const double x) const {
        // return the extreme values if a point outside the interpolation
        // region is asked for
        if (x <= x_min_) {
            return y_min_;
        }
        if (x >= x_max_) {
            return y_max_;
        }

        // find the index to interpolate inside of
        size_t idx = 0;
        for (size_t i = 0; i < n_pts_; i++) {
            if (x_(i) >= x) {
                idx = i - 1;
                break;
            }
        }
        double dx_plus = x_(idx + 1) - x;
        double dx_minus = x - x_(idx);
        double delta_xi = x_(idx + 1) - x_(idx);
        double a = (y_dash_dash_(idx) * dx_plus * dx_plus * dx_plus +
                    y_dash_dash_(idx + 1) * dx_minus * dx_minus * dx_minus) /
                   (6 * delta_xi);
        double b = (y_(idx) / delta_xi - y_dash_dash_(idx) * delta_xi / 6) * dx_plus;
        double c =
            (y_(idx + 1) / delta_xi - y_dash_dash_(idx + 1) * delta_xi / 6) * dx_minus;
        return a + b + c;
    }

private:
    // The points to interpolate
    Kokkos::View<double*> x_;
    Kokkos::View<double*> y_;

    // the second derivatives to make evaluating the interpolation efficient
    Kokkos::View<double*> y_dash_dash_;

    double x_min_;
    double x_max_;
    double y_min_;
    double y_max_;

    size_t n_pts_;
};
