#include <doctest/doctest.h>
#include "cfl.h"

CflSchedule::CflSchedule(json schedule) 
    : times_(schedule.at("times")),
      cfls_(schedule.at("cfls"))
{}

double CflSchedule::eval(double t) {
    if (t <= times_[0]) {return cfls_[0];}

    size_t n = times_.size();
    if (t >= times_[n-1]){return cfls_[n-1];}

    double cfl;
    for (size_t i = 0; i < times_.size(); i++) {
        if (t > times_[i] && t < times_[i+1]) {
            double frac = (t - times_[i]) / (times_[i+1] - times_[i]); 
            printf("i = %lu, frac = %f\n", i, frac);
            cfl = cfls_[i] + frac * (cfls_[i+1] - cfls_[i]);
            return cfl;
        }
    }
    throw std::runtime_error("Shouldn't reach here");
}

TEST_CASE("CflSchedule") {
    json times = {0.0, 1.0, 2.0};
    json cfls = {0.1, 0.2, 0.5};
    json schedule_json = {{"times", times}, {"cfls", cfls}};

    CflSchedule schedule {schedule_json};
    CHECK(schedule.eval(0.0) == doctest::Approx(0.1));
    CHECK(schedule.eval(0.5) == doctest::Approx(0.15));
    CHECK(schedule.eval(1.5) == doctest::Approx(0.35));
    CHECK(schedule.eval(2.5) == doctest::Approx(0.5));
}
