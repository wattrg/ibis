#ifndef LIMITER_H
#define LIMITER_H

#include <gas/flow_state.h>
#include <grid/grid.h>
#include <util/vector3.h>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

enum class Limiters { None, BarthJespersen };

namespace Ibis {

Limiters string_to_limiter(std::string name);

}

template <typename T>
struct LimiterValues {
    LimiterValues() {}

    LimiterValues(int num_cells) {
        p = Field<T>("Limiter::p", num_cells);
        rho = Field<T>("Limiter::rho", num_cells);
        vx = Field<T>("Limiter::vx", num_cells);
        vy = Field<T>("Limiter::vy", num_cells);
        vz = Field<T>("Limiter::vz", num_cells);
    }

    Field<T> p;
    Field<T> rho;
    Field<T> vx;
    Field<T> vy;
    Field<T> vz;
};

template <typename T, class SubView>
void barth_jespersen(const SubView values, Field<T>& limits,
                     const Cells<T>& cells, const Interfaces<T>& faces,
                     Vector3s<T>& grad) {
    Kokkos::parallel_for(
        "Limiter::barth_jesperson", cells.num_valid_cells(),
        KOKKOS_LAMBDA(const int cell_i) {
            T Ui = values(cell_i);
            T U_min = Ui;
            T U_max = Ui;
            for (size_t j = 0; j < cells.neighbour_cells(cell_i).size(); j++) {
                int neighbour_cell = cells.neighbour_cells(cell_i, j);
                U_min = Kokkos::min(U_min, values(neighbour_cell));
                U_max = Kokkos::max(U_max, values(neighbour_cell));
            }

            T phi = 1.0;
            auto face_ids = cells.faces().face_ids(cell_i);
            for (size_t j = 0; j < face_ids.size(); j++) {
                int i_face = face_ids(j);
                T dx = faces.centre().x(i_face);
                T dy = faces.centre().y(i_face);
                T dz = faces.centre().z(i_face);
                T delta_2 = grad.x(cell_i) * dx + grad.y(cell_i) * dy +
                            grad.z(cell_i) * dz;
                int sign_delta_2 = (delta_2 > 0) - (delta_2 < 0);
                delta_2 = sign_delta_2 * (Kokkos::abs(delta_2) + 1e-16);
                if (sign_delta_2 > 0) {
                    phi = Kokkos::min(phi, (U_max - Ui) / delta_2);
                } else if (sign_delta_2 < 0) {
                    phi = Kokkos::min(phi, (U_min - Ui) / delta_2);
                }
            }
            limits(cell_i) = phi;
        });
}

template <typename T>
class Limiter {
public:
    Limiter() {}

    Limiter(Limiters limiter) : limiter_(limiter) {}

    Limiter(json config) : limiter_() {
        limiter_ = Ibis::string_to_limiter(config.at("limiter"));
    }

    template <class SubView>
    void calculate_limiters(const SubView values, Field<T>& limits,
                            const Cells<T>& cells, const Interfaces<T>& faces,
                            Vector3s<T>& grad) {
        switch (limiter_) {
            case Limiters::None:
                break;
            case Limiters::BarthJespersen:
                barth_jespersen(values, limits, cells, faces, grad);
        }
    }

    KOKKOS_INLINE_FUNCTION
    bool enabled() const { return limiter_ != Limiters::None; }

private:
    Limiters limiter_;
};

#endif
