#ifndef GRID_H
#define GRID_H

#include "grid_io.h"
#include "../../util/src/id.h"

template <typename T>
struct GridBlock {
public:
    GridBlock(const GridIO &grid_io); 

private:
    Vertices<T> _vertices;
};

#endif
