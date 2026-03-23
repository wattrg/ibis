#include <solvers/high_order_blending.h>
#include <spdlog/spdlog.h>
#include <iostream>

#include <stdexcept>

ConstantOrder::ConstantOrder(Ibis::real order) {
    if (order >= 1.0 && order <= 2.0) {
        limiter_value_ = order - 1;
    } else {
        throw std::runtime_error("ConstantOrder: Invalid order");
    }
}

Ibis::real ConstantOrder::eval_global_limiter() { return limiter_value_; }

LinearResidualBasedHighOrderBlending::LinearResidualBasedHighOrderBlending(
    Ibis::real start_blending_residual, Ibis::real stop_blending_residual, Ibis::real maximum_decrease) {
    ln_start_blending_residual_ = log(start_blending_residual);
    ln_stop_blending_residual_ = log(stop_blending_residual);
    maximum_decrease_ = maximum_decrease;
    ln_residual_ = 0.0;
    previous_limiter_ = 0.0;
}

LinearResidualBasedHighOrderBlending::LinearResidualBasedHighOrderBlending(json config) {
    Ibis::real start_blending_residual = config.at("start_blending_residual");
    Ibis::real stop_blending_residual = config.at("stop_blending_residual");
    ln_start_blending_residual_ = log(start_blending_residual);
    ln_stop_blending_residual_ = log(stop_blending_residual);
    ln_residual_ = 0.0;
    maximum_decrease_ = config.at("maximum_decrease");
    previous_limiter_ = 0.0;
}

void LinearResidualBasedHighOrderBlending::set_residual(Ibis::real residual) {
    ln_residual_ = log(residual);
}

Ibis::real LinearResidualBasedHighOrderBlending::eval_global_limiter() {
    if (ln_residual_ > ln_start_blending_residual_) {
        previous_limiter_ = 0.0;
    } else if (ln_residual_ > ln_stop_blending_residual_) {
        Ibis::real limiter = (ln_residual_ - ln_start_blending_residual_) /
                             (ln_stop_blending_residual_ - ln_start_blending_residual_);
        previous_limiter_ = std::max(previous_limiter_ / maximum_decrease_, limiter);
    } else {
        previous_limiter_ = 1.0;
    }
    return previous_limiter_;
}

std::unique_ptr<HighOrderBlendingSchedule> make_high_order_blending_schedule(
    json config) {
    std::string type = config.at("type");
    if (type == "constant") {
        Ibis::real order = config.at("order");
        return std::make_unique<ConstantOrder>(order);
    }
    if (type == "linear_residual_based_blending") {
        return std::make_unique<LinearResidualBasedHighOrderBlending>(config);
    }
    spdlog::error("Invalid high order blending type {}", type);
    throw new std::runtime_error("Invalid order blending type");
}
