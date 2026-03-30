#include <finite_volume/limiter.h>
#include <spdlog/spdlog.h>
#include <util/numeric_types.h>

#include <stdexcept>

template <typename T>
std::unique_ptr<Limiter<T>> make_limiter(json config) {
    std::string limiter_type = config.at("type");
    if (limiter_type == "barth_jespersen") {
        Ibis::real epsilon = config.at("epsilon");
        return std::unique_ptr<Limiter<T>>(new BarthJespersen<T>(epsilon));
    } else if (limiter_type == "venkat") {
        Ibis::real K = config.at("K");
        return std::unique_ptr<Limiter<T>>(new Venkat<T>(K));
    } else if (limiter_type == "unlimited") {
        return std::unique_ptr<Limiter<T>>(new Unlimited<T>());
    } else {
        spdlog::error("Unknown limiter {}", limiter_type);
        throw new std::runtime_error("Unknown limiter");
    }
}
template std::unique_ptr<Limiter<Ibis::real>> make_limiter<Ibis::real>(json);
template std::unique_ptr<Limiter<Ibis::dual>> make_limiter<Ibis::dual>(json);

template <typename T>
void BarthJespersen<T>::calculate_limiters(const Ibis::SubArray2D<T> values,
                                           Field<T>& limits, const Cells<T>& cells,
                                           const Interfaces<T>& faces,
                                           Vector3s<T>& grad) {
    Ibis::real epsilon = epsilon_;
    Kokkos::parallel_for(
        "Limiter::barth_jesperson", cells.num_valid_cells(),
        KOKKOS_LAMBDA(const size_t cell_i) {
            T Ui = values(cell_i);
            T U_min = Ui;
            T U_max = Ui;
            T U_avg = T(0.0);
            size_t num_neighbours = cells.neighbour_cells(cell_i).size();
            for (size_t j = 0; j < num_neighbours; j++) {
                size_t neighbour_cell = cells.neighbour_cells(cell_i, j);
                U_min = Ibis::min(U_min, values(neighbour_cell));
                U_max = Ibis::max(U_max, values(neighbour_cell));
                U_avg += values(neighbour_cell);
            }
            U_avg /= (num_neighbours + 1);

            T phi = 1.0;
            T x = cells.centroids().x(cell_i);
            T y = cells.centroids().y(cell_i);
            T z = cells.centroids().z(cell_i);
            auto face_ids = cells.faces().face_ids(cell_i);
            for (size_t j = 0; j < face_ids.size(); j++) {
                int i_face = face_ids(j);
                T dx = faces.centre().x(i_face) - x;
                T dy = faces.centre().y(i_face) - y;
                T dz = faces.centre().z(i_face) - z;
                T delta_2 =
                    grad.x(cell_i) * dx + grad.y(cell_i) * dy + grad.z(cell_i) * dz;
                int sign_delta_2 = (delta_2 > 0) - (delta_2 < 0);
                delta_2 = sign_delta_2 * (Ibis::abs(delta_2) + epsilon * U_avg);
                if (sign_delta_2 > 0) {
                    phi = Ibis::min(phi, (U_max - Ui) / delta_2);
                } else if (sign_delta_2 < 0) {
                    phi = Ibis::min(phi, (U_min - Ui) / delta_2);
                }
            }
            limits(cell_i) = phi;
        });
}
template class BarthJespersen<Ibis::real>;
template class BarthJespersen<Ibis::dual>;

template <typename T>
void Venkat<T>::calculate_limiters(const Ibis::SubArray2D<T> values,
                                   Field<T>& limits, const Cells<T>& cells,
                                   const Interfaces<T>& faces,
                                   Vector3s<T>& grad) {
    Ibis::real K = K_;
    Kokkos::parallel_for(
        "Limiter::venkat", cells.num_valid_cells(),
        KOKKOS_LAMBDA(const size_t cell_i) {
            T Ui = values(cell_i);
            T U_min = Ui;
            T U_max = Ui;
            T U_avg = T(0.0);
            size_t num_neighbours = cells.neighbour_cells(cell_i).size();
            for (size_t j = 0; j < num_neighbours; j++) {
                size_t neighbour_cell = cells.neighbour_cells(cell_i, j);
                U_min = Ibis::min(U_min, values(neighbour_cell));
                U_max = Ibis::max(U_max, values(neighbour_cell));
                U_avg += values(neighbour_cell);
            }
            U_avg /= (num_neighbours + 1);
            T e2 = K * Ibis::cbrt(cells.volume(cell_i));
            e2 = e2 * e2 * e2 * U_avg;
            T phi = 1.0;
            T x = cells.centroids().x(cell_i);
            T y = cells.centroids().y(cell_i);
            T z = cells.centroids().z(cell_i);
            auto face_ids = cells.faces().face_ids(cell_i);
            for (size_t j = 0; j < face_ids.size(); j++) {
                int i_face = face_ids(j);
                T dx = faces.centre().x(i_face) - x;
                T dy = faces.centre().y(i_face) - y;
                T dz = faces.centre().z(i_face) - z;
                T delta_2 =
                    grad.x(cell_i) * dx + grad.y(cell_i) * dy + grad.z(cell_i) * dz;
                T delta_2_2 = delta_2 * delta_2;
                if (delta_2 > 0) {
                    T delta_1_max = U_max - values(cell_i);
                    T delta_1_max2 = delta_1_max * delta_1_max;
                    T num = (delta_1_max2 + e2) * delta_2 + 2 * delta_2_2 * delta_1_max;
                    T den = delta_1_max2 + 2 * delta_2_2 + delta_1_max * delta_2 + e2;
                    phi = Ibis::min(phi, 1 / delta_2 * (num / den));
                } else if (delta_2 < 0) {
                    T delta_1_min = U_min - values(cell_i);
                    T delta_1_min2 = delta_1_min * delta_1_min;
                    T num = (delta_1_min2 + e2) * delta_2 + 2 * delta_2_2 * delta_1_min;
                    T den = delta_1_min2 + 2 * delta_2_2 + delta_1_min * delta_2 + e2;
                    phi = Ibis::min(phi, 1 / delta_2 * (num / den));
                }
            }
            limits(cell_i) = phi;
        });
}
template class Venkat<Ibis::real>;
template class Venkat<Ibis::dual>;

template <typename T>
void Unlimited<T>::calculate_limiters(const Ibis::SubArray2D<T> values, Field<T>& limits,
                                      const Cells<T>& cells, const Interfaces<T>& faces,
                                      Vector3s<T>& grad) {
    (void)values;
    (void)limits;
    (void)cells;
    (void)faces;
    (void)grad;
}
template class Unlimited<Ibis::real>;
template class Unlimited<Ibis::dual>;
