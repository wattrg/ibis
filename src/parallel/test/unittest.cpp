#define DOCTEST_CONFIG_IMPLEMENT

#include <doctest/doctest.h>
#include <Kokkos_Core.hpp>

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    doctest::Context ctx;

    ctx.applyCommandLine(argc, argv);
    int res = ctx.run();

    Kokkos::finalize();

    return res;
}
