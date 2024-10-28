#include <spdlog/spdlog.h>
#include <finite_volume/grid_motion_driver.h>
#include <finite_volume/rigid_body_translation.h>
#include <finite_volume/shock_fitting.h>

template <typename T>
std::shared_ptr<GridMotionDriver<T>> build_grid_motion_driver(json config) {
    if (!config.at("enabled")) {
        spdlog::error("Attemping to build grid motion driver when grid motion is disabled");
        throw new std::runtime_error("Building grid motion driver with grid motion disabled");
    }

    std::string type = config.at("type");
    if (type == "rigid_body_translation") {
        return std::shared_ptr<GridMotionDriver<T>>(new RigidBodyTranslation<T>(config));
    }
    else if (type == "shock_fitting") {
      return std::shared_ptr<GridMotionDriver<T>>(new ShockFitting<T>(config));
    }
    else {
      spdlog::error("Unknown grid motion driver {}", type);
      throw new std::runtime_error("Unkown grid motion driver");
    }
}

template std::shared_ptr<GridMotionDriver<Ibis::real>> build_grid_motion_driver<Ibis::real>(json config);
template std::shared_ptr<GridMotionDriver<Ibis::dual>> build_grid_motion_driver<Ibis::dual>(json config);
