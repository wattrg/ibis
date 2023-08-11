#ifndef INTERFACE_H
#define INTERFACE_H

#include "../../util/src/id.h"
#include "../../util/src/field.h"
#include "../../util/src/vector3.h"
#include <unordered_map>

template <typename T>
struct Interfaces {

private:
    // the id's of the vertices forming each interface
    Id _vertex_ids;

    // geometric data
    Aeolus::Field<T> _area; 
    Aeolus::Vector3s<T> _norm;
    Aeolus::Vector3s<T> _tan1;
    Aeolus::Vector3s<T> _tan2;
    Aeolus::Vector3s<T> _centre;
};

// Efficient look-up of interface ID 
// from the index of the vertices
// forming the interface
struct InterfaceLookup {
public:
    InterfaceLookup();

    int insert(std::vector<int> vertex_ids);
    bool contains(std::vector<int> vertex_ids);
    int id(std::vector<int> vertex_ids); 

private:
    std::unordered_map<std::string, int> _hash_map;

    std::string _hash(std::vector<int> vertex_ids);
    bool _contains(std::string hash);
};

#endif
