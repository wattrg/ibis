#include <doctest/doctest.h>
#include <triangular_solver_kokkos_kernels/triangular_solver_kokkos_kernels.h>
#include <util/types.h>

#ifndef DOCTEST_CONFIG_DISABLE
TEST_CASE("triangular_solver_kokkos_kernels") {
    using array_type =
        Ibis::Array1D<int, Ibis::DefaultArrayLayout, Ibis::DefaultMemSpace>;
    using value_type =
        Ibis::Array1D<double, Ibis::DefaultArrayLayout, Ibis::DefaultMemSpace>;

    array_type L_row_map("L_row_map", 5);
    array_type L_entries("L_entries", 7);
    value_type L_values("L_values", 7);
    auto L_row_map_h = Kokkos::create_mirror_view(L_row_map);
    auto L_entries_h = Kokkos::create_mirror_view(L_entries);
    auto L_values_h = Kokkos::create_mirror_view(L_values);
    L_row_map_h(0) = 0;
    L_row_map_h(1) = 1;
    L_row_map_h(2) = 3;
    L_row_map_h(3) = 5;
    L_row_map_h(4) = 7;
    Kokkos::deep_copy(L_row_map, L_row_map_h);

    L_entries_h(0) = 0;
    L_entries_h(1) = 0;
    L_entries_h(2) = 1;
    L_entries_h(3) = 1;
    L_entries_h(4) = 2;
    L_entries_h(5) = 2;
    L_entries_h(6) = 3;
    Kokkos::deep_copy(L_entries, L_entries_h);

    L_values_h(0) = 1.0;
    L_values_h(1) = -0.25;
    L_values_h(2) = 1.0;
    L_values_h(3) = -0.258065;
    L_values_h(4) = 1.0;
    L_values_h(5) = -0.25833;
    L_values_h(6) = 1.0;
    Kokkos::deep_copy(L_values, L_values_h);
    Ibis::CrsGraph<int, int> L_graph(L_row_map, L_entries);
    Ibis::CrsMatrix<int, int, double> L(L_graph, L_values);

    array_type U_row_map("U_row_map", 5);
    array_type U_entries("U_entries", 7);
    value_type U_values("U_values", 7);
    auto U_row_map_h = Kokkos::create_mirror_view(U_row_map);
    auto U_entries_h = Kokkos::create_mirror_view(U_entries);
    auto U_values_h = Kokkos::create_mirror_view(U_values);
    U_row_map_h(0) = 0;
    U_row_map_h(1) = 2;
    U_row_map_h(2) = 4;
    U_row_map_h(3) = 6;
    U_row_map_h(4) = 7;
    Kokkos::deep_copy(U_row_map, U_row_map_h);

    U_entries_h(0) = 0;
    U_entries_h(1) = 1;
    U_entries_h(2) = 1;
    U_entries_h(3) = 2;
    U_entries_h(4) = 2;
    U_entries_h(5) = 3;
    U_entries_h(6) = 3;
    Kokkos::deep_copy(U_entries, U_entries_h);

    U_values_h(0) = 4.0;
    U_values_h(1) = -0.5;
    U_values_h(2) = 3.875;
    U_values_h(3) = -0.5;
    U_values_h(4) = 3.87097;
    U_values_h(5) = -0.5;
    U_values_h(6) = 2.87083;
    Kokkos::deep_copy(U_values, U_values_h);
    Ibis::CrsGraph<int, int> U_graph(U_row_map, U_entries);
    Ibis::CrsMatrix<int, int, double> U(U_graph, U_values);

    using SolverType = KokkosKernels_SparseTriangularSolver<
        Ibis::DefaultExecSpace, Ibis::DefaultMemSpace, Ibis::DefaultArrayLayout>;
    auto L_solver = SolverType(L, Ibis::TriangularMatrixType::LOWER);
    auto U_solver = SolverType(U, Ibis::TriangularMatrixType::UPPER);

    Ibis::Vector<Ibis::real> x("x", 4);
    Ibis::Vector<Ibis::real> y("y", 4);
    Ibis::Vector<Ibis::real> b("b", 4);

    L_solver.solve(L, b, y);
    U_solver.solve(U, y, x);

    auto x_h = x.host_mirror();
    auto y_h = y.host_mirror();
    x_h.deep_copy_space(x);
    y_h.deep_copy_space(y);
}
#endif
