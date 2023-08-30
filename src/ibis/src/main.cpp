#include <iostream>
#include <Kokkos_Core.hpp>
#include "../../gas/src/gas_state.h"

typedef Kokkos::View<double*[3]> FieldTest;

int test(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);

    // write hello a few times
    Kokkos::parallel_for("Hello world", 15, KOKKOS_LAMBDA (const int i){
        printf("Hello from i = %i\n", i);
    });

    // add up square numbers in parallel
    int sum = 0;
    const int n = 10;
    Kokkos::parallel_reduce("reduction", n, KOKKOS_LAMBDA (const int i, int& lsum) {
        lsum += i*i;
    }, sum);
    printf("The sum of the first %i square numbers is %i\n", n-1, sum);

    {
        // play around with a views
        FieldTest a ("A", n);
        Kokkos::parallel_for("View", n, KOKKOS_LAMBDA (const int i) {
            a(i, 0) = 1.0 * i;
            a(i, 1) = 1.0 * i * i;
            a(i, 2) = 1.0 * i * i * i;
        });
        double view_sum = 0.0;
        Kokkos::parallel_reduce ("Reduction", n, KOKKOS_LAMBDA (const int i, double& update) {
            update += a(i, 0) * a(i,1) / (a(i,2) + 0.1);
        }, view_sum);
        printf("Result: %f\n", view_sum);
    }

    {
        GasStates<double> gs = GasStates<double>(5);
        std::cout << "Built some gas states!" << std::endl;
        printf("initial gs energy = %f\n", gs.energy(2));
    }

    Kokkos::finalize();
    return 0;
}
