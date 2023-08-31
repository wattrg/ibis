import json
import os
from enum import Enum

IBIS = os.environ.get("IBIS")

validation_errors = []

class ValidationException(Exception):
    pass

def read_defaults(file_name):
    with open(f"{IBIS}/resources/defaults/{file_name}", "r") as defaults:
        defaults = json.load(defaults)
    return defaults

class FluxCalculator(Enum):
    Hanel = "hanel"

def string_to_flux_calculator(string):
    if string == FluxCalculator.Hanel.value:
        return FluxCalculator.Hanel
    validation_errors.append(ValidationException(f"Unknown flux calculator {string}"))

class ConvectiveFlux:
    _json_values = ["flux_calculator", "reconstruction_order"]
    __slots__ = _json_values
    _defaults_file = "convective_flux.json"
    
    def __init__(self):
        json_data = read_defaults(self._defaults_file) 
        for key in self._json_values:
            setattr(self, key, json_data[key])
        self.flux_calculator = string_to_flux_calculator(self.flux_calculator)

    def validate(self):
        if type(self.flux_calculator) != FluxCalculator:
            self.flux_calculator = string_to_flux_calculator(self.flux_calculator)
        if self.reconstruction_order not in (1, 2):
            validation_errors.append(ValidationException(f"reconstruction order {self.reconstruction_order} not supported"))

    def as_dict(self):
        dictionary = {}
        for key in self._json_values:
            if key == "flux_calculator":
                dictionary[key] = self.flux_calculator.value
            else:
                dictionary[key] = getattr(self, key)
        return dictionary

class Config:
    __slots__ = ["convective_flux"]

    def __init__(self):
        self.convective_flux = ConvectiveFlux()
    
    def validate(self):
        for setting in self.__slots__:
            getattr(self, setting).validate()
        if validation_errors:
            raise ValidationException(validation_errors)

    def write(self, directory, file_name):
        json_values = {}
        for setting in self.__slots__:
            json_values[setting] = getattr(self, setting).as_dict()

        with open(f"{directory}/{file_name}", "w") as f:
            json.dump(json_values, f, indent=4)


def main(file_name):
    # make the config directory
    directories = read_defaults("directories.json")
    config_dir = directories["config_dir"]
    config_status = directories["config_status"]
    if not os.path.exists(config_dir):
        os.mkdir(config_dir)

    # write a file saying the config is incomplete
    with open(f"{config_dir}/{config_status}", "w") as status:
        status.write("False")

    config = Config()
    namespace = {
        "config": config,
        "ConvectiveFlux": ConvectiveFlux,
        "FluxCalculator": FluxCalculator
    }

    with open(file_name, "r") as f:
        exec(f.read(), namespace)

    config.validate()
    config.write(directories["config_dir"], directories["config_file"])

    with open(f"{config_dir}/{config_status}", "w") as status:
        status.write("True")

