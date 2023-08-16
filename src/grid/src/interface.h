#ifndef INTERFACE_H
#define INTERFACE_H

#include <unordered_map>
#include "../../util/src/id.h"
#include "../../util/src/field.h"
#include "../../util/src/vector3.h"

template <typename T> struct Interfaces;

template <typename T>
struct InterfaceView {
public:
    InterfaceView(Interfaces<T> *interfaces, int id) 
        : _interfaces(interfaces), _id(id) {}

    inline auto vertex_ids() {return _interfaces->vertex_ids()[_id];}
    inline T& area() const {return _interfaces->area()[_id];}
    inline T& norm() const {return _interfaces->norm()[_id];}
    inline T& tan1() const {return _interfaces->tan1()[_id];}
    inline T& tan2() const {return _interfaces->tan2()[_id];}

private:
    Interfaces<T> * _interfaces;
    int _id;
};

template <typename T>
struct Interfaces {
public:
    Interfaces () {}

    Interfaces(IdConstructor ids) : _vertex_ids(Id(ids)) {}

    bool operator == (const Interfaces &other) const {
        return _vertex_ids == other._vertex_ids;
    }

    inline InterfaceView<T> operator[] (const int i) {
        assert(i < size());
        return InterfaceView<T>(this, i);
    }

    inline Id &vertex_ids() {return _vertex_ids;}
    inline Field<T> &area() const {return _area;}
    inline Field<T> &norm() const {return _norm;}
    inline Field<T> &tan1() const {return _tan1;}
    inline Field<T> &tan2() const {return _tan2;}
    inline Vector3s<T> &centre() const {return _centre;}

    inline int size() const {return _vertex_ids.size();}

private:
    // the id's of the vertices forming each interface
    Id _vertex_ids;

    // geometric data
    Field<T> _area; 
    Vector3s<T> _norm;
    Vector3s<T> _tan1;
    Vector3s<T> _tan2;
    Vector3s<T> _centre;
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
