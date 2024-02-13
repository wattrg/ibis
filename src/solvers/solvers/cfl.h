#ifndef CFL_H
#define CFL_H

#include <nlohmann/json.hpp>
using json = nlohmann::json;

class CflSchedule {
public:
    CflSchedule() {}

    CflSchedule(json schedule);

    double eval(double t);

private:
    std::vector<double> times_;
    std::vector<double> cfls_;
};

#endif
