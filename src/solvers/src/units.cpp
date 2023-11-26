#include "units.h"

Units::Units(double time_, double mass_, double length_, double current_, 
      double temp_, double amount_, double luminosity_) :
    time(time_),
    mass(mass_),
    length(length_),
    current(current_),
    temp(temp_),
    amount(amount_),
    luminosity(luminosity_)
{}

Units::Units(json config) 
    : Units(config.at("time"), 
            config.at("mass"), 
            config.at("length"),
            config.at("current"),
            config.at("temp"),
            config.at("amount"),
            config.at("luminosity"))
{}
