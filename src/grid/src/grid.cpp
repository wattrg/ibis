#include <doctest/doctest.h>
#include "grid.h"

TEST_CASE("build grid block") {
    GridBlock block = GridBlock<double>("../test/grid.su2");
}
