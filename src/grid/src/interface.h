#ifndef INTERFACE_H
#define INTERFACE_H

#include "../../util/src/id.h"
#include "../../util/src/field.h"
#include "../../util/src/vector3.h"

struct Interfaces {

private:
    // the id's of the vertices forming each interface
    Id _vertex_ids;

    // geometric data
    Aeolus::Field _area; 
    Aeolus::Vector3s _norm;
    Aeolus::Vector3s _tan1;
    Aeolus::Vector3s _tan2;
    Aeolus::Vector3s _centre;
};

#endif
