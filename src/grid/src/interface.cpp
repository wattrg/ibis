#include <algorithm>
#include <functional>
#include <doctest/doctest.h>
#include "interface.h"

InterfaceLookup::InterfaceLookup() {
    _hash_map = std::unordered_map<std::string, int> {};
}

int InterfaceLookup::insert(std::vector<int> vertex_ids) {
    std::string hash = _hash(vertex_ids);
    if (_contains(hash)){
        return _hash_map[hash];
    }
    int id = _hash_map.size();
    _hash_map.insert({hash, id});
    return id;
}

bool InterfaceLookup::contains(std::vector<int> vertex_ids) {
    std::string hash = _hash(vertex_ids);
    return _contains(hash);
}

int InterfaceLookup::id(std::vector<int> vertex_ids) {
    std::string hash = _hash(vertex_ids);
    if (_contains(hash)) {
        return _hash_map[hash];
    }
    return -1;
}

bool InterfaceLookup::_contains(std::string hash) {
    return _hash_map.find(hash) != _hash_map.end();
}

std::string InterfaceLookup::_hash(std::vector<int> vertex_ids) {
    std::sort(vertex_ids.begin(), vertex_ids.end(), std::greater<int>());
    std::string hash_value = "";
    for (unsigned int i = 0; i < vertex_ids.size(); i++) {
        hash_value.append(std::to_string(vertex_ids[i]));
        hash_value.append(",");
    }
    return hash_value;
}

TEST_CASE("interface look up contains") {
    InterfaceLookup x;
    x.insert(std::vector<int> {0, 1});
    x.insert(std::vector<int> {1, 5});
    x.insert(std::vector<int> {5, 4});
    x.insert(std::vector<int> {5, 1});

    CHECK(x.contains(std::vector<int> {1, 0}));
    CHECK(x.contains(std::vector<int> {6, 1}) == false);
}

TEST_CASE("interface look up id 1") {
    InterfaceLookup x;
    x.insert(std::vector<int> {0, 1});
    x.insert(std::vector<int> {1, 5});
    x.insert(std::vector<int> {5, 4});
    x.insert(std::vector<int> {5, 1});

    CHECK(x.id(std::vector<int> {5, 1}) == 1);
    CHECK(x.id(std::vector<int> {1, 5}) == 1);
}

TEST_CASE("interface look up id 2") {
    InterfaceLookup x;
    x.insert(std::vector<int> {0, 1});
    x.insert(std::vector<int> {1, 5});
    x.insert(std::vector<int> {5, 4});
    x.insert(std::vector<int> {5, 1});

    CHECK(x.id(std::vector<int> {6, 1}) == -1);
}

TEST_CASE("interface look up") {
    InterfaceLookup x;
    x.insert(std::vector<int> {0, 1});
    x.insert(std::vector<int> {1, 5});
    x.insert(std::vector<int> {5, 4});
    x.insert(std::vector<int> {4, 0});
    x.insert(std::vector<int> {1, 2});
    x.insert(std::vector<int> {2, 6});
    x.insert(std::vector<int> {6, 5});
    x.insert(std::vector<int> {5, 1});
    x.insert(std::vector<int> {2, 3});
    x.insert(std::vector<int> {3, 7});
    x.insert(std::vector<int> {7, 6});
    x.insert(std::vector<int> {6, 2});

    CHECK(x.id(std::vector<int> {0, 1}) == 0);
    CHECK(x.id(std::vector<int> {1, 5}) == 1);
    CHECK(x.id(std::vector<int> {5, 4}) == 2);
    CHECK(x.id(std::vector<int> {4, 0}) == 3);
    CHECK(x.id(std::vector<int> {1, 2}) == 4);
    CHECK(x.id(std::vector<int> {2, 6}) == 5);
    CHECK(x.id(std::vector<int> {6, 5}) == 6);
    CHECK(x.id(std::vector<int> {2, 3}) == 7);
    CHECK(x.id(std::vector<int> {3, 7}) == 8);
    CHECK(x.id(std::vector<int> {7, 6}) == 9);
}
