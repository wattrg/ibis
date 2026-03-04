#define DOCTEST_CONFIG_IMPLEMENT

#include <doctest/doctest.h>

#include <Kokkos_Core.hpp>

int main(int argc, char* argv[]) {
    doctest::Context ctx;
    Kokkos::initialize(argc, argv);
    ctx.applyCommandLine(argc, argv);
    int res = ctx.run();
    Kokkos::finalize();
    return res;
}
