#ifndef UNITS_H
#define UNITS_H

#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct Units {
public:
    Units(double time_, double mass_, double length_, double current_, 
          double temp_, double amount_, double luminosity_);

    Units (json config);

public:
    double time;
    double mass;
    double length;
    double current;
    double temp;
    double amount;
    double luminosity;
};

#endif
