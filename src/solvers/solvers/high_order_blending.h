#ifndef HIGH_ORDER_BLENDING
#define HIGH_ORDER_BLENDING

#include <util/numeric_types.h>
#include <memory>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class HighOrderBlendingSchedule {
public:
    HighOrderBlendingSchedule() {}

    virtual ~HighOrderBlendingSchedule() {}

    virtual Ibis::real eval_global_limiter() = 0;
};

class ConstantOrder : public HighOrderBlendingSchedule {
public:
    ConstantOrder() {}

    ConstantOrder(Ibis::real order);

    Ibis::real eval_global_limiter();

private:
    Ibis::real limiter_value_;
};

class LinearResidualBasedHighOrderBlending : public HighOrderBlendingSchedule {
public:
    LinearResidualBasedHighOrderBlending() {}

    LinearResidualBasedHighOrderBlending(Ibis::real start_blending_residual,
                                   Ibis::real stop_blending_residual);

    LinearResidualBasedHighOrderBlending(json config);

    void set_residual(Ibis::real);

    Ibis::real eval_global_limiter();

private:
    Ibis::real ln_start_blending_residual_;
    Ibis::real ln_stop_blending_residual_;
    Ibis::real ln_residual_;
};

std::unique_ptr<HighOrderBlendingSchedule> make_high_order_blending_schedule(json config);

#endif
