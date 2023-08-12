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
        for (int i = 0; i < ids.size(); i++) {
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

    Id(Kokkos::View<int*> ids, Kokkos::View<int*> offsets)
        : _ids(ids), _offsets(offsets){}

    Id(std::vector<int> ids, std::vector<int> offsets);

    Id(IdConstructor constructor) 
        : Id(constructor.ids(), constructor.offsets()) {}

    inline
    auto operator [] (const int i) const {
        // return the slice of _ids corresponding 
        // to the object at index i
        int first = _offsets(i);
        int last = _offsets(i+1);
        return Kokkos::subview(_ids, std::make_pair(first, last));
    }

private:
    Kokkos::View<int*> _ids;
    Kokkos::View<int*> _offsets;
};


#endif
