#ifndef IBIS_CONFIG_H
#define IBIS_CONFIG_H

#include <nlohmann/json.hpp>

using json = nlohmann::json;

json read_directories();
json read_config(json& directories);

#endif
