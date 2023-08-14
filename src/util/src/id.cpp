#include <doctest/doctest.h>
#include "id.h"

Id::Id(std::vector<int> ids, std::vector<int> offsets) {
    _ids = Kokkos::View<int*> ("id", static_cast<int>(ids.size()));
    _offsets = Kokkos::View<int*> ("offset", static_cast<int>(offsets.size()));

    for (unsigned int i = 0; i < ids.size(); i++) {
        _ids(i) = ids[i];
    }

    for (unsigned int i = 0; i < offsets.size(); i++) {
        _offsets(i) = offsets[i];
    }
}

TEST_CASE("id") {
    Id ids = Id(std::vector<int> {1,2,3,4,5,6}, std::vector<int> {0, 3});
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
    Id id (idc);

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
