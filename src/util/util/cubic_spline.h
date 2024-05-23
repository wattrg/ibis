#include <Kokkos_Core.hpp>
#include <vector>

class CubicSpline {
public:
    CubicSpline() {}
    CubicSpline(std::vector<double> x, std::vector<double> y);

    KOKKOS_INLINE_FUNCTION
    double eval(const double x) const {
        // find the index to interpolate inside of
        size_t idx = 0;
        for (size_t i = 0; i < n_pts_; i++) {
            if (x_(i) >= x) {
                idx = i - 1;
            }
        }

        double dx_plus = x_(idx + 1) - x;
        double dx_minus = x - x_(idx);
        double delta_xi = x_(idx + 1) - x_(idx);
        double a = (y_dash_dash_(idx) * dx_plus * dx_plus * dx_plus +
                    y_dash_dash_(idx + 1) * dx_minus * dx_minus * dx_minus) /
                   (6 * delta_xi);
        double b = (y_(idx) / delta_xi - y_dash_dash_(idx) * delta_xi / 6) * dx_plus;
        double c = (y_(idx + 1) / delta_xi - y_dash_dash_(idx + 1) * delta_xi / 6) * dx_minus;
        return a + b + c;
    }

private:
    // The points to interpolate
    Kokkos::View<double*> x_;
    Kokkos::View<double*> y_;

    // the second derivatives to make evaluating the interpolation efficient
    Kokkos::View<double*> y_dash_dash_;

    size_t n_pts_;
};
