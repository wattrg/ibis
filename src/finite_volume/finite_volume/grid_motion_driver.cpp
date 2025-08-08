#include <finite_volume/grid_motion_driver.h>
#include <finite_volume/rigid_body_translation.h>
#include <finite_volume/shock_fitting.h>
#include <spdlog/spdlog.h>

template <typename T, class MemModel>
std::shared_ptr<GridMotionDriver<T, MemModel>> build_grid_motion_driver(const GridBlock<MemModel, T>& grid,
                                                              json config) {
    if (!config.at("enabled")) {
        spdlog::error(
            "Attemping to build grid motion driver when grid motion is disabled");
        throw new std::runtime_error(
            "Building grid motion driver with grid motion disabled");
    }

    std::string type = config.at("type");
    if (type == "rigid_body_translation") {
        return std::shared_ptr<GridMotionDriver<T, MemModel>>(new RigidBodyTranslation<T, MemModel>(config));
    } else if (type == "boundary_interpolation") {
        return std::shared_ptr<GridMotionDriver<T, MemModel>>(new ShockFitting<T, MemModel>(grid, config));
    } else {
        spdlog::error("Unknown grid motion driver {}", type);
        throw new std::runtime_error("Unkown grid motion driver");
    }
}

template std::shared_ptr<GridMotionDriver<Ibis::real, SharedMem>>
build_grid_motion_driver<Ibis::real, SharedMem>(const GridBlock<SharedMem, Ibis::real>&, json);
template std::shared_ptr<GridMotionDriver<Ibis::dual, SharedMem>>
build_grid_motion_driver<Ibis::dual, SharedMem>(const GridBlock<SharedMem, Ibis::dual>&, json config);
