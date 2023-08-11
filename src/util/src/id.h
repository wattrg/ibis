#ifndef ID_H
#define ID_H

#include <vector>
#include <Kokkos_Core.hpp>

struct Id {
public:
    Id(Kokkos::View<int*> ids, Kokkos::View<int*> offsets)
        : _ids(ids), _offsets(offsets){}

    Id(std::vector<int> ids, std::vector<int> offsets);

    inline
    auto operator [] (const int i) const {
        int first = _offsets(i);
        int last = _offsets(i+1);
        return Kokkos::subview(_ids, std::make_pair(first, last));
    }

private:
    Kokkos::View<int*> _ids;
    Kokkos::View<int*> _offsets;
};

#endif
