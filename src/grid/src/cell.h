#ifndef CELL_H
#define CELL_H

#include "../../util/src/id.h"
#include "../../util/src/field.h"

template <typename T>
struct Cells {
public:

private:
    Id _interface_ids;
    Id _vertex_ids;
    Aeolus::Field<T> _volume;
    Aeolus::Field<int> _outsign;
};

#endif
