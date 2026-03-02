#include <doctest/doctest.h>
#include <ilu_kokkos_kernels/ilu_kokkos_kernels.h>
#include <util/types.h>


TEST_CASE("ilu_kokkos_kernels") {
    using array_type = Ibis::Array1D<int, Ibis::DefaultArrayLayout, Ibis::DefaultMemSpace>;
    array_type A_rowmap("rowmap", 5);
    array_type A_entries("entries", 10);
    Ibis::Array1D<double, Ibis::DefaultArrayLayout, Ibis::DefaultMemSpace> A_values("values", 10);
    A_rowmap(0) = 0;
    A_rowmap(1) = 2;
    A_rowmap(2) = 5;
    A_rowmap(3) = 8;
    A_rowmap(4) = 10;

    A_entries(0) = 0;
    A_entries(1) = 1;
    A_entries(2) = 0;
    A_entries(3) = 1;
    A_entries(4) = 2;
    A_entries(5) = 1;
    A_entries(6) = 2;
    A_entries(7) = 3;
    A_entries(8) = 2;
    A_entries(9) = 3;

    A_values(0) = 4.0;
    A_values(1) = -0.5;
    A_values(2) = -1.0;
    A_values(3) = 4.0;
    A_values(4) = -0.5;
    A_values(5) = -1.0;
    A_values(6) = 4.0;
    A_values(7) = -0.5;
    A_values(8) = -1.0;
    A_values(9) = 3.0;

    Ibis::CrsGraph<int, int> A_graph(A_rowmap, A_entries);
    Ibis::CrsMatrix<int, int, double> A(A_graph, A_values);
    auto ilu = KokkosKernels_ILU<Ibis::DefaultExecSpace, Ibis::DefaultMemSpace, Ibis::DefaultArrayLayout>(A_graph, 0);

    ilu.numeric_phase(A);

    CHECK(ilu.L.graph.row_map(0) == 0);
    CHECK(ilu.L.graph.row_map(1) == 1);
    CHECK(ilu.L.graph.row_map(2) == 3);
    CHECK(ilu.L.graph.row_map(3) == 5);
    CHECK(ilu.L.graph.row_map(4) == 7);

    CHECK(ilu.U.graph.row_map(0) == 0);
    CHECK(ilu.U.graph.row_map(1) == 2);
    CHECK(ilu.U.graph.row_map(2) == 4);
    CHECK(ilu.U.graph.row_map(3) == 6);
    CHECK(ilu.U.graph.row_map(4) == 7);

    CHECK(ilu.L.graph.entries(0) == 0);
    CHECK(ilu.L.graph.entries(1) == 0);
    CHECK(ilu.L.graph.entries(2) == 1);
    CHECK(ilu.L.graph.entries(3) == 1);
    CHECK(ilu.L.graph.entries(4) == 2);
    CHECK(ilu.L.graph.entries(5) == 2);
    CHECK(ilu.L.graph.entries(6) == 3);

    CHECK(ilu.U.graph.entries(0) == 0);
    CHECK(ilu.U.graph.entries(1) == 1);
    CHECK(ilu.U.graph.entries(2) == 1);
    CHECK(ilu.U.graph.entries(3) == 2);
    CHECK(ilu.U.graph.entries(4) == 2);
    CHECK(ilu.U.graph.entries(5) == 3);
    CHECK(ilu.U.graph.entries(6) == 3);

    CHECK(ilu.L.values(0) == doctest::Approx(1.0));
    CHECK(ilu.L.values(1) == doctest::Approx(-0.25));
    CHECK(ilu.L.values(2) == doctest::Approx(1.0));
    CHECK(ilu.L.values(3) == doctest::Approx(-0.258065));
    CHECK(ilu.L.values(4) == doctest::Approx(1.0));
    CHECK(ilu.L.values(5) == doctest::Approx(-0.25833));
    CHECK(ilu.L.values(6) == doctest::Approx(1.0));

    CHECK(ilu.U.values(0) == doctest::Approx(4.0));
    CHECK(ilu.U.values(1) == doctest::Approx(-0.5));
    CHECK(ilu.U.values(2) == doctest::Approx(3.875));
    CHECK(ilu.U.values(3) == doctest::Approx(-0.5));
    CHECK(ilu.U.values(4) == doctest::Approx(3.87097));
    CHECK(ilu.U.values(5) == doctest::Approx(-0.5));
    CHECK(ilu.U.values(6) == doctest::Approx(2.87083));

}

