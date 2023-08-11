#include "id.h"

Id::Id(std::vector<int> ids, std::vector<int> offsets) {
    _ids = Kokkos::View<int*> ("id", static_cast<int>(ids.size()));
    _offsets = Kokkos::View<int*> ("offset", static_cast<int>(offsets.size()));

    for (int i = 0; i < ids.size(); i++) {
        _ids(i) = ids[i];
    }

    for (int i = 0; i < offsets.size(); i++) {
        _offsets(i) = offsets[i];
    }
}
