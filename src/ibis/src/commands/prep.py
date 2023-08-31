import json
import os

ibis = os.environ.get("IBIS")

def read_defaults(file_name):
    with open(f"{ibis}/resources/defaults/{file_name}", "r") as defaults:
        defaults = json.load(defaults)
    return defaults

class ConvectiveFlux:
    _json_values = ["flux_calculator", "reconstruction_order"]
    __slots__ = _json_values
    _defaults_file = "convective_flux.json"
    
    def __init__(self):
        json_data = read_defaults(self._defaults_file) 
        for key in self._json_values:
            setattr(self, key, json_data[key])

    def as_dict(self):
        dictionary = {}
        for key in self._json_values:
            dictionary[key] = getattr(self, key)
        return dictionary

class Config:
    __slots__ = ["convective_flux"]

    def __init__(self):
        self.convective_flux = ConvectiveFlux()

    def write(self, directory, file_name):
        if not os.path.exists(directory):
            os.mkdir(directory)

        json_values = {}
        json_values["convective_flux"] = self.convective_flux.as_dict()
        with open(f"{directory}/{file_name}", "w") as f:
            json.dump(json_values, f, indent=4)


def main(file_name):
    config = Config()
    namespace = {
        "config": config,
        "ConvectiveFlux": ConvectiveFlux,
    }
    directories = read_defaults("directories.json")

    with open(file_name, "r") as f:
        exec(f.read(), namespace)
    config.write(directories["config_dir"], directories["config_file"])

