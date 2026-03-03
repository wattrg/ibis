#include <doctest/doctest.h>
#include <ilu_kokkos_kernels/ilu_kokkos_kernels.h>
#include <util/types.h>

#include "Kokkos_DynRankView.hpp"

TEST_CASE("ilu_kokkos_kernels") {
    using array_type =
        Ibis::Array1D<int, Ibis::DefaultArrayLayout, Ibis::DefaultMemSpace>;
    array_type A_rowmap("rowmap", 5);
    array_type A_entries("entries", 10);
    Ibis::Array1D<double, Ibis::DefaultArrayLayout, Ibis::DefaultMemSpace> A_values(
        "values", 10);
    auto A_rowmap_h = Kokkos::create_mirror_view(A_rowmap);
    auto A_entries_h = Kokkos::create_mirror_view(A_entries);
    auto A_values_h = Kokkos::create_mirror_view(A_values);
    A_rowmap_h(0) = 0;
    A_rowmap_h(1) = 2;
    A_rowmap_h(2) = 5;
    A_rowmap_h(3) = 8;
    A_rowmap_h(4) = 10;
    Kokkos::deep_copy(A_rowmap, A_rowmap_h);

    A_entries_h(0) = 0;
    A_entries_h(1) = 1;
    A_entries_h(2) = 0;
    A_entries_h(3) = 1;
    A_entries_h(4) = 2;
    A_entries_h(5) = 1;
    A_entries_h(6) = 2;
    A_entries_h(7) = 3;
    A_entries_h(8) = 2;
    A_entries_h(9) = 3;
    Kokkos::deep_copy(A_entries, A_entries_h);

    A_values_h(0) = 4.0;
    A_values_h(1) = -0.5;
    A_values_h(2) = -1.0;
    A_values_h(3) = 4.0;
    A_values_h(4) = -0.5;
    A_values_h(5) = -1.0;
    A_values_h(6) = 4.0;
    A_values_h(7) = -0.5;
    A_values_h(8) = -1.0;
    A_values_h(9) = 3.0;
    Kokkos::deep_copy(A_values, A_values_h);

    Ibis::CrsGraph<int, int> A_graph(A_rowmap, A_entries);
    Ibis::CrsMatrix<int, int, double> A(A_graph, A_values);
    auto ilu = KokkosKernels_ILU<Ibis::DefaultExecSpace, Ibis::DefaultMemSpace,
                                 Ibis::DefaultArrayLayout>(A_graph, 0);

    ilu.numeric_phase(A);

    auto L_h = ilu.L.host_mirror();
    L_h.deep_copy(ilu.L);
    auto U_h = ilu.U.host_mirror();
    U_h.deep_copy(ilu.U);
    CHECK(L_h.num_rows() == 4);
    CHECK(L_h.graph.row_map.size() == 5);
    CHECK(L_h.graph.entries.size() == 7);
    CHECK(L_h.values.size() == 7);

    CHECK(U_h.num_rows() == 4);
    CHECK(U_h.graph.row_map.size() == 5);
    CHECK(U_h.graph.entries.size() == 7);
    CHECK(U_h.values.size() == 7);

    CHECK(L_h.graph.row_map(0) == 0);
    CHECK(L_h.graph.row_map(1) == 1);
    CHECK(L_h.graph.row_map(2) == 3);
    CHECK(L_h.graph.row_map(3) == 5);
    CHECK(L_h.graph.row_map(4) == 7);

    CHECK(U_h.graph.row_map(0) == 0);
    CHECK(U_h.graph.row_map(1) == 2);
    CHECK(U_h.graph.row_map(2) == 4);
    CHECK(U_h.graph.row_map(3) == 6);
    CHECK(U_h.graph.row_map(4) == 7);

    CHECK(L_h.graph.entries(0) == 0);
    CHECK(L_h.graph.entries(1) == 0);
    CHECK(L_h.graph.entries(2) == 1);
    CHECK(L_h.graph.entries(3) == 1);
    CHECK(L_h.graph.entries(4) == 2);
    CHECK(L_h.graph.entries(5) == 2);
    CHECK(L_h.graph.entries(6) == 3);

    CHECK(U_h.graph.entries(0) == 0);
    CHECK(U_h.graph.entries(1) == 1);
    CHECK(U_h.graph.entries(2) == 1);
    CHECK(U_h.graph.entries(3) == 2);
    CHECK(U_h.graph.entries(4) == 2);
    CHECK(U_h.graph.entries(5) == 3);
    CHECK(U_h.graph.entries(6) == 3);

    CHECK(L_h.values(0) == doctest::Approx(1.0));
    CHECK(L_h.values(1) == doctest::Approx(-0.25));
    CHECK(L_h.values(2) == doctest::Approx(1.0));
    CHECK(L_h.values(3) == doctest::Approx(-0.258065));
    CHECK(L_h.values(4) == doctest::Approx(1.0));
    CHECK(L_h.values(5) == doctest::Approx(-0.25833));
    CHECK(L_h.values(6) == doctest::Approx(1.0));

    CHECK(U_h.values(0) == doctest::Approx(4.0));
    CHECK(U_h.values(1) == doctest::Approx(-0.5));
    CHECK(U_h.values(2) == doctest::Approx(3.875));
    CHECK(U_h.values(3) == doctest::Approx(-0.5));
    CHECK(U_h.values(4) == doctest::Approx(3.87097));
    CHECK(U_h.values(5) == doctest::Approx(-0.5));
    CHECK(U_h.values(6) == doctest::Approx(2.87083));
}
