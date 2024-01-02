#include <spdlog/spdlog.h>
#include <finite_volume/limiter.h>
#include <stdexcept>

namespace Ibis {

Limiters string_to_limiter(std::string name) {
    if (name == "none") return Limiters::None;
    if (name == "barth_jespersen") return Limiters::BarthJespersen;
    else {
        spdlog::error("Invalid limiter {}", name);
        throw std::runtime_error("Invalid limiter");
    }
}

}
