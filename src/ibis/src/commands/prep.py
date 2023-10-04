import json
import os
import shutil
from enum import Enum

from ibis_py_utils import read_defaults, FlowState

validation_errors = []

class ValidationException(Exception):
    pass


class FluxCalculator(Enum):
    Hanel = "hanel"

class Solver(Enum):
    RungeKutta = "runge_kutta"

def string_to_flux_calculator(string):
    if string == FluxCalculator.Hanel.value:
        return FluxCalculator.Hanel
    validation_errors.append(ValidationException(f"Unknown flux calculator {string}"))

def string_to_solver(string):
    if string == Solver.RungeKutta.value:
        return Solver.RungeKutta
    validation_errors.append(ValidationException(f"Unknown solver {string}"))

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


class Block:
    def __init__(self, file_name, initial_condition, boundaries):
        self._initial_condition = initial_condition
        self._block = file_name
        self.number_cells = 0
        self.number_vertices = 0
        self.boundaries = boundaries
        self._read_file(file_name)

    def _read_file(self, file_name):
        extension = file_name.split(".")[-1]
        if extension == "su2":
            self._read_su2_grid(file_name)
        else:
            raise Exception(f"Unknown grid format: {extension}")

    def _read_su2_grid(self, file_name):
        with open(file_name, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("NDIME"):
                    self.dim = int(line.split("=")[-1].strip())
                elif line.startswith("NPOIN"):
                    self.number_vertices = int(line.split("=")[-1].strip())
                elif line.startswith("NELEM"):
                    self.number_cells = int(line.split("=")[-1].strip())
                elif line.startswith("NMARK"):
                    self.number_boundaries = int(line.split("=")[-1].strip())

    def validate(self):
        if not self._block:
            validation_errors.append(
                ValidationException("No grid blocks specified")
            )

    def write(self, grid_directory, flow_directory):
        if not os.path.exists(grid_directory):
            os.mkdir(grid_directory)
        if not os.path.exists(flow_directory):
            os.mkdir(flow_directory)
        ic_path = f"{flow_directory}/0000"
        if not os.path.exists(ic_path):
            os.mkdir(ic_path)

        # write the grid
        shutil.copy(self._block, f"{grid_directory}/block_{0:04}.su2")
        
        # write the initial condition
        ic_directory = f"{flow_directory}/{0:04}"
        temp = open(f"{ic_directory}/T", "w")
        pressure = open(f"{ic_directory}/p", "w")
        vx = open(f"{ic_directory}/vx", "w")
        vy = open(f"{ic_directory}/vy", "w")
        if self.dim == 3:
            vz = open(f"{ic_directory}/vz", "w")

        if type(self._initial_condition) == FlowState:
            for _ in range(self.number_cells):
                temp.write(str(self._initial_condition.T) + "\n")
                pressure.write(str(self._initial_condition.p) + "\n")
                vx.write(str(self._initial_condition.vx) + "\n")
                vy.write(str(self._initial_condition.vy) + "\n")
                if self.dim == 3:
                    vz.write(str(self._initial_condition.vz) + "\n")

        temp.close()
        pressure.close()
        vx.close()
        vy.close()
        if self.dim == 3:
            vz.close()

    def as_dict(self):
        dictionary = {"boundaries": {}}
        dictionary["dimensions"] = self.dim
        for key in self.boundaries:
            dictionary["boundaries"][key] = self.boundaries[key].as_dict()
        return dictionary

class BoundaryCondition:
    def __init__(self, pre_reconstruction, ghost_cells=True):
        self._pre_reconstruction = pre_reconstruction
        self.ghost_cells = ghost_cells

    def as_dict(self):
        dictionary = {}
        pre_reco_dict = []
        for pre_reco in self._pre_reconstruction:
            pre_reco_dict.append(pre_reco.as_dict())
        dictionary["pre_reconstruction"] = pre_reco_dict
        dictionary["ghost_cells"] = self.ghost_cells
        return dictionary

class FlowStateCopy:
    def __init__(self, flow_state):
        self.flow_state = flow_state

    def as_dict(self):
        return {"type": "flow_state_copy", "flow_state": self.flow_state.as_dict()}

def supersonic_inflow(inflow):
    return BoundaryCondition(
        pre_reconstruction=[FlowStateCopy(inflow)]
    )


class RungeKutta:
    _json_values = ["cfl", "max_time", "max_step", "print_frequency", 
                    "plot_frequency", "plot_every_n_steps"]
    _defaults_file = "runge_kutta.json"
    _name = Solver.RungeKutta.value
    __slots__ = _json_values

    def __init__(self):
        json_data = read_defaults(self._defaults_file)
        for key in json_data:
            setattr(self, key, json_data[key])

    def as_dict(self):
        dictionary = {"name": self._name}
        for key in self._json_values:
            dictionary[key] = getattr(self, key)
        return dictionary

    def validate(self):
        return

def make_default_solver():
    default_solver_name = read_defaults("config.json")["solver"] 
    default_solver = string_to_solver(default_solver_name)
    if default_solver == Solver.RungeKutta:
        return RungeKutta()
    validation_errors.append(
        ValidationException(f"Unknown default solver {default_solver_name}")
    )

class Config:
    _json_values = ["convective_flux", "solver", "grid"]
    __slots__ = _json_values

    def __init__(self):
        self.convective_flux = ConvectiveFlux()
        self.solver = make_default_solver()
    
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
        flow_directory = directories["flow_dir"]
        self.grid.write(grid_directory, flow_directory)


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
        "Block": Block,
        "Solver": Solver,
        "FlowState": FlowState,
        "RungeKutta": RungeKutta,
        "SupersonicInflow": supersonic_inflow
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
