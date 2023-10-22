#include <doctest/doctest.h>
#include "id.h"


TEST_CASE("id") {
    Id<>::mirror_type ids(std::vector<int> {1,2,3,4,5,6}, std::vector<int> {0, 3});
    auto sub_id = ids[0];
    CHECK(sub_id(0) == 1);
    CHECK(sub_id(1) == 2);
    CHECK(sub_id(2) == 3);

    sub_id = ids[1];
    CHECK(sub_id(0) == 4);
    CHECK(sub_id(1) == 5);
    CHECK(sub_id(2) == 6);
}


TEST_CASE("id constructioin") {
    IdConstructor idc;
    idc.push_back(std::vector<int> {3, 2, 5});
    idc.push_back(std::vector<int> {1, 3});
    idc.push_back(std::vector<int> {2, 8, 9});

    Id<>::mirror_type id (idc);

    auto sub_id = id[0];
    CHECK(sub_id(0) == 3);
    CHECK(sub_id(1) == 2);
    CHECK(sub_id(2) == 5);
    
    sub_id = id[1];
    CHECK(sub_id(0) == 1);
    CHECK(sub_id(1) == 3);

    sub_id = id[2];
    CHECK(sub_id(0) == 2);
    CHECK(sub_id(1) == 8);
    CHECK(sub_id(2) == 9);
}
