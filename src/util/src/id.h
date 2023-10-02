#ifndef ID_H
#define ID_H

#include <vector>
#include <Kokkos_Core.hpp>

struct IdConstructor {
public:
    IdConstructor() {
        _ids = std::vector<int> {};
        _offsets = std::vector<int> {0};
    }

    void push_back(std::vector<int> ids) {
        for (unsigned int i = 0; i < ids.size(); i++) {
            _ids.push_back(ids[i]);
        }
        _offsets.push_back(_offsets.back() + ids.size());
    }

    std::vector<int> ids() const {return _ids;}
    std::vector<int> offsets() const {return _offsets;}

private:
    std::vector<int> _ids;
    std::vector<int> _offsets;
};


struct Id {
public:
    Id() {}

    Id(Kokkos::View<int*> ids, Kokkos::View<int*> offsets);

    Id(std::vector<int> ids, std::vector<int> offsets);

    Id(IdConstructor constructor) 
        : Id(constructor.ids(), constructor.offsets()) {}

    Id clone();

    Id clone_offsets();

    KOKKOS_INLINE_FUNCTION
    auto operator [] (const int i) const {
        // return the slice of _ids corresponding 
        // to the object at index i
        int first = _offsets(i);
        int last = _offsets(i+1);
        return Kokkos::subview(_ids, std::make_pair(first, last));
    }

    KOKKOS_INLINE_FUNCTION
    auto operator [] (const int i) {
        // return the slice of _ids corresponding 
        // to the object at index i
        int first = _offsets(i);
        int last = _offsets(i+1);
        return Kokkos::subview(_ids, std::make_pair(first, last));
    }

    KOKKOS_INLINE_FUNCTION
    int size() const {return _offsets.extent(0)-1;}

    bool operator == (const Id &other) const {
        for (unsigned int i = 0; i < _ids.extent(0); i++) {
            if (_ids(i) != other._ids(i)) return false;
        }
        for (unsigned int i = 0; i < _offsets.extent(0); i++) {
            if (_offsets(i) != other._offsets(i)) return false;
        }
        return true;
    }

    Kokkos::View<int*> ids() const {return _ids;}
    Kokkos::View<int*> offsets() const {return _offsets;}

private:
    Kokkos::View<int*> _ids;
    Kokkos::View<int*> _offsets;
};


#endif
