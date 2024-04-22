#ifndef CFL_H
#define CFL_H

#include <memory>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

class CflSchedule {
public:
    CflSchedule() {}

    virtual ~CflSchedule() {}

    virtual double eval(double t) = 0;
};

class ConstantSchedule : public CflSchedule {
public:
    ConstantSchedule() {}

    ConstantSchedule(double cfl);

    double eval(double t);

private:
    double cfl_;
};

class LinearSchedule : public CflSchedule {
public:
    LinearSchedule() {}

    LinearSchedule(json schedule);

    double eval(double t);

private:
    std::vector<double> times_;
    std::vector<double> cfls_;
};

std::unique_ptr<CflSchedule> make_cfl_schedule(json config);

#endif
