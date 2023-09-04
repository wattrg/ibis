import json
import os
import shutil
from enum import Enum

from ibis_py_utils import read_defaults

validation_errors = []

class ValidationException(Exception):
    pass


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
            validation_errors.append(
                ValidationException(
                    f"reconstruction order {self.reconstruction_order} not supported"
                )
            )

    def as_dict(self):
        dictionary = {}
        for key in self._json_values:
            if key == "flux_calculator":
                dictionary[key] = self.flux_calculator.value
            else:
                dictionary[key] = getattr(self, key)
        return dictionary

class Grid:
    def __init__(self):
        self._blocks = []

    def add_block(self, file_name):
        self._blocks.append(file_name)

    def validate(self):
        if not self._blocks:
            validation_errors.append(ValidationException("No grid blocks specified"))

    def write(self, grid_directory):
        if not os.path.exists(grid_directory):
            os.mkdir(grid_directory)

        for i, file in enumerate(self._blocks):
            shutil.copy(file, f"{grid_directory}/block_{i:04}.su2")

class Config:
    _json_values = ["convective_flux"]
    __slots__ = _json_values + ["grid"]

    def __init__(self):
        self.convective_flux = ConvectiveFlux()
        self.grid = Grid()
    
    def validate(self):
        for setting in self.__slots__:
            getattr(self, setting).validate()
        if validation_errors:
            raise ValidationException(validation_errors)

    def write(self, directories):
        # directories to place things in
        config_directory = directories["config_dir"]
        config_file = directories["config_file"]
        config_file = f"{config_directory}/{config_file}"

        # extract all the values to go in the json config file
        json_values = {}
        for setting in self._json_values:
            json_values[setting] = getattr(self, setting).as_dict()

        # write the json config file
        with open(config_file, "w") as f:
            json.dump(json_values, f, indent=4)

        # write the grid files
        grid_directory = directories["grid_dir"]
        self.grid.write(f"{grid_directory}")


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

    # build a namespace to execute the user supplied preparation script with
    config = Config()
    namespace = {
        "config": config,
        "ConvectiveFlux": ConvectiveFlux,
        "FluxCalculator": FluxCalculator,
        "Grid": Grid,
    }

    # run the user supplied script
    with open(file_name, "r") as f:
        exec(f.read(), namespace)

    # validate the configuration supplied by the user, and hopefully write
    # the configuration to file.
    config.validate()
    config.write(directories)
    with open(f"{config_dir}/{config_status}", "w") as status:
        status.write("True")
