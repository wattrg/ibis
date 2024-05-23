#include <Kokkos_Core.hpp>
#include <vector>

class CubicSpline {
public:
    CubicSpline(std::vector<double> x, std::vector<double> y);

    KOKKOS_FUNCTION
    double eval(const double x) const;

private:
    // The points to interpolate
    Kokkos::View<double*> x_;
    Kokkos::View<double*> y_;

    // the second derivatives to make evaluating the interpolation efficient
    Kokkos::View<double*> y_dash_dash_;

    size_t n_pts_;
};
