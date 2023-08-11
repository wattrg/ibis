#ifndef FIELD_H
#define FIELD_H

#include <Kokkos_Core.hpp>

namespace Aeolus {

template <typename T>
using Field = Kokkos::View<T*>;

}

#endif
