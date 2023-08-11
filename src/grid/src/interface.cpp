#include "interface.h"
#include <algorithm>
#include <functional>

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
    for (int i = 0; i < vertex_ids.size(); i++) {
        hash_value.append(std::to_string(vertex_ids[i]));
        hash_value.append(",");
    }
    return hash_value;
}
