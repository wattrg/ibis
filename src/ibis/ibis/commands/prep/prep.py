import json
import os
import shutil
from enum import Enum

from ibis_py_utils import (
    ConstantPrandtlNumber,
    SutherlandViscosity,
    TransportPropertyModel,
    read_defaults,
    FlowState,
    GasModel
)

from python_api import (
    PyAusmdv,
    PyHanel,
    PyLdfss2,
    GasState,
    PyIdealGas
)

validation_errors = []


class ValidationException(Exception):
    pass


class Solver(Enum):
    RungeKutta = "runge_kutta"


class Limiter(Enum):
    BarthJespersen = "barth_jespersen"


def string_to_solver(string):
    if string == Solver.RungeKutta.value:
        return Solver.RungeKutta
    validation_errors.append(ValidationException(f"Unknown solver {string}"))


def string_to_limiter(string):
    if string == Limiter.BarthJespersen.value:
        return Limiter.BarthJespersen
    if string == "none":
        return None
    else:
        validation_errors.append(
            ValidationException(f"Unknown limiter {string}")
        )


def string_from_limiter(limiter):
    if limiter:
        return limiter.value
    return "none"


class FluxCalculator:
    def _read_defaults(self):
        json_data = read_defaults(DEFAULTS_DIRECTORY, self._defaults_file)
        for key in self._json_values:
            setattr(self, key, json_data[key])


class Hanel(FluxCalculator):
    _defaults_file = "hanel.json"
    _json_values = []

    def __init__(self):
        self._read_defaults()
        self._flux_calc = PyHanel()

    def as_dict(self):
        return {"type": self._flux_calc.name()}


class Ausmdv(FluxCalculator):
    _defaults_file = "ausmdv.json"
    _json_values = []

    def __init__(self):
        self._read_defaults()
        self._flux_calc = PyAusmdv()

    def as_dict(self):
        self._read_defaults()
        return {"type": self._flux_calc.name()}


class Ldfss2(FluxCalculator):
    _defaults_file = "ldfss2.json"
    _json_values = ["delta"]

    def __init__(self, delta=None):
        self._read_defaults()
        self._flux_calc = PyLdfss2()

        if delta:
            self.delta = delta

    def as_dict(self):
        return {"type": self._flux_calc.name(), "delta": self.delta}


def string_to_flux_calc(name):
    if name == "hanel":
        return Hanel()
    elif name == "ausmdv":
        return Ausmdv()
    elif name == "ldfss2":
        return Ldfss2()
    else:
        validation_errors.append(
            ValidationException(f"Unknown flux calculator {name}")
        )


class ConvectiveFlux:
    _json_values = ["flux_calculator", "reconstruction_order", "limiter"]
    __slots__ = _json_values
    _defaults_file = "convective_flux.json"

    def __init__(self, **kwargs):
        json_data = read_defaults(DEFAULTS_DIRECTORY,
                                  self._defaults_file)
        for key in self._json_values:
            if key == "flux_calculator":
                self.flux_calculator = string_to_flux_calc(
                    json_data["flux_calculator"]
                )
            setattr(self, key, json_data[key])

        for key in kwargs:
            setattr(self, key, kwargs[key])

        if type(self.limiter) is not Limiter:
            self.limiter = string_to_limiter(self.limiter)

    def validate(self):
        if self.reconstruction_order not in (1, 2):
            validation_errors.append(
                ValidationException(
                    f"reconstruction order {self.reconstruction_order}"
                    " not supported"
                )
            )

    def as_dict(self):
        dictionary = {}
        for key in self._json_values:
            if key == "flux_calculator":
                dictionary[key] = self.flux_calculator.as_dict()
            elif key == "limiter":
                dictionary[key] = string_from_limiter(self.limiter)
            else:
                dictionary[key] = getattr(self, key)
        return dictionary


class ViscousFlux:
    _json_values = ["enabled", "signal_factor"]
    __slots__ = _json_values
    _defaults_file = "viscous_flux.json"

    def __init__(self, **kwargs):
        json_data = read_defaults(DEFAULTS_DIRECTORY,
                                  self._defaults_file)
        for key in self._json_values:
            setattr(self, key, json_data[key])

        for key in kwargs:
            setattr(self, key, kwargs[key])

    def validate(self):
        if self.signal_factor < 0:
            validation_errors.append(ValidationException(
                f"Invalid signal_factor {self.signal_factor}")
            )

    def as_dict(self):
        dictionary = {}
        for key in self._json_values:
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
        meta_data = open(f"{ic_directory}/meta_data.json", "w")
        times = open(f"{flow_directory}/flows", "w")

        if type(self._initial_condition) is FlowState:
            for _ in range(self.number_cells):
                temp.write(f"{self._initial_condition.gas.T:.16e}\n")
                pressure.write(f"{self._initial_condition.gas.p:.16e}\n")
                vx.write(f"{self._initial_condition.vel.x:.16e}\n")
                vy.write(f"{self._initial_condition.vel.y:.16e}\n")
                if self.dim == 3:
                    vz.write(f"{self._initial_condition.vel.z:.16e}\n")
        json.dump({"time": 0.0}, meta_data, indent=4)
        times.write("0000\n")

        temp.close()
        pressure.close()
        vx.close()
        vy.close()
        if self.dim == 3:
            vz.close()
        meta_data.close()
        times.close()

    def as_dict(self):
        dictionary = {"boundaries": {}}
        dictionary["dimensions"] = self.dim
        for key in self.boundaries:
            dictionary["boundaries"][key] = self.boundaries[key].as_dict()
        return dictionary


class BoundaryCondition:
    def __init__(self, pre_reconstruction, pre_viscous_grad, ghost_cells=True):
        self._pre_reconstruction = pre_reconstruction
        self._pre_viscous_grad = pre_viscous_grad
        self.ghost_cells = ghost_cells

    def as_dict(self):
        dictionary = {}
        pre_reco_dict = []
        for pre_reco in self._pre_reconstruction:
            pre_reco_dict.append(pre_reco.as_dict())
        dictionary["pre_reconstruction"] = pre_reco_dict

        pre_viscous_grad_dict = []
        for pre_viscous_grad in self._pre_viscous_grad:
            pre_viscous_grad_dict.append(pre_viscous_grad.as_dict())
        dictionary["pre_viscous_grad"] = pre_viscous_grad_dict

        dictionary["ghost_cells"] = self.ghost_cells
        return dictionary


class _FlowStateCopy:
    def __init__(self, flow_state):
        self.flow_state = flow_state

    def as_dict(self):
        return {
            "type": "flow_state_copy",
            "flow_state": self.flow_state.as_dict()
        }


class _InternalCopy:
    def as_dict(self):
        return {"type": "internal_copy"}


class _InternalCopyReflectNormal:
    def as_dict(self):
        return {"type": "internal_copy_reflect_normal"}


class _InternalVelCopyReflect:
    def as_dict(self):
        return {"type": "internal_vel_copy_reflect"}


def supersonic_inflow(inflow):
    return BoundaryCondition(
        pre_reconstruction=[_FlowStateCopy(inflow)],
        pre_viscous_grad=[]
    )


def supersonic_outflow():
    return BoundaryCondition(
        pre_reconstruction=[_InternalCopy()],
        pre_viscous_grad=[]
    )


def slip_wall():
    return BoundaryCondition(
        pre_reconstruction=[_InternalCopyReflectNormal()],
        pre_viscous_grad=[]
    )


def adiabatic_no_slip_wall():
    return BoundaryCondition(
        pre_reconstruction=[_InternalCopyReflectNormal()],
        pre_viscous_grad=[_InternalVelCopyReflect()]
    )


class CflSchedule:
    def as_dict(self):
        pass


class ConstantCfl(CflSchedule):
    _type = "constant"

    def __init__(self, cfl):
        self._cfl = cfl

    def as_dict(self):
        return {"type": self._type, "value": self._cfl}


class LinearInterpolateCfl(CflSchedule):
    _type = "linear_interpolate"

    def __init__(self, cfls):
        self._times = []
        self._cfls = []
        for time, cfl in cfls:
            self._times.append(time)
            self._cfls.append(cfl)

    def as_dict(self):
        return {
            "type": self._type,
            "times": self._times,
            "cfls": self._cfls
        }


def make_cfl_schedule(config):
    if type(config) is float:
        return ConstantCfl(config)

    cfl_type = config["type"]
    if cfl_type == "constant":
        return ConstantCfl(config["value"])
    elif cfl_type == "linear_interpolate":
        cfls = []
        for time, cfl in zip(config["times"], config["cfls"]):
            cfls.append((time, cfl))
        return LinearInterpolateCfl(cfls)
    else:
        validation_errors.append(
            ValidationException(f"Unkown cfl schedule {cfl_type}")
        )


class RungeKutta:
    _json_values = ["cfl", "max_time", "max_step", "print_frequency",
                    "plot_frequency", "plot_every_n_steps", "dt_init"]
    _defaults_file = "runge_kutta.json"
    _name = Solver.RungeKutta.value
    __slots__ = _json_values

    def __init__(self, **kwargs):
        json_data = read_defaults(DEFAULTS_DIRECTORY,
                                  self._defaults_file)
        for key in json_data:
            setattr(self, key, json_data[key])

        for key in kwargs:
            setattr(self, key, kwargs[key])

    def as_dict(self):
        dictionary = {"name": self._name}
        for key in self._json_values:
            if key == "cfl" and type(self.cfl) is not CflSchedule:
                self.cfl = make_cfl_schedule(self.cfl).as_dict()
            dictionary[key] = getattr(self, key)
        return dictionary

    def validate(self):
        return


def make_default_solver():
    default_solver_name = read_defaults(DEFAULTS_DIRECTORY,
                                        "config.json")["solver"]
    default_solver = string_to_solver(default_solver_name)
    if default_solver == Solver.RungeKutta:
        return RungeKutta()
    validation_errors.append(
        ValidationException(f"Unknown default solver {default_solver_name}")
    )


def load_species_data(species):
    with open(f"{SHARE_DIRECTORY}/species_database/{species}.json") as f:
        species_data = json.load(f)
    return species_data


class IdealGas(GasModel):
    _type = "ideal_gas"

    def __init__(self, **args):
        if "R" in args:
            self._gas_model = PyIdealGas(args["R"])
        elif "species" in args:
            species = args["species"]
            if type(species) is str:
                self._species = species
            if type(species) is list or type(species) is tuple:
                if len(species) != 1:
                    raise Exception("Multiple species not supported yet")
                self._species = species[0]

            species_data = load_species_data(self._species)
            R = species_data["thermo"]["R"]
            self._gas_model = PyIdealGas(R)

    def as_dict(self):
        return {
            "type": self._type,
            "R": self._gas_model.R(),
            "Cv": self._gas_model.Cv(),
            "Cp": self._gas_model.Cp(),
            "gamma": self._gas_model.gamma()
        }


def default_gas_model():
    defaults = read_defaults(DEFAULTS_DIRECTORY, "config.json")
    default_gas_model = defaults["gas_model"]
    if default_gas_model["model"] == "ideal_gas":
        return IdealGas(species=default_gas_model["species"])


def build_viscosity_model(gas_model):
    gas_model_type = gas_model.type()
    with open(f"{DEFAULTS_DIRECTORY}/transport_properties.json") as f:
        models = json.load(f)
    viscosity_model = models["viscosity"][gas_model_type]
    species_data = load_species_data(gas_model.species())["transport"]
    if viscosity_model == "sutherland":
        species_data = species_data["sutherland"]
        mu_0 = species_data["mu_0"]
        T_0 = species_data["T_0"]
        T_s = species_data["T_s"]
        return SutherlandViscosity(mu_0, T_0, T_s)
    else:
        validation_errors.append(
            ValidationException(f"Unknown default viscosity "
                                f"model {viscosity_model}")
        )


def build_thermal_conductivity_model(gas_model):
    gas_model_type = gas_model.type()
    with open(f"{DEFAULTS_DIRECTORY}/transport_properties.json") as f:
        models = json.load(f)
    thermal_conductivity = models["thermal_conductivity"][gas_model_type]
    species_data = load_species_data(gas_model.species())["transport"]
    if thermal_conductivity == "constant_prandtl":
        Pr = species_data["prandtl"]
        return ConstantPrandtlNumber(Pr)
    else:
        validation_errors.append(
            ValidationException("Unknown default thermal conductivity "
                                f"model {thermal_conductivity}")
        )


def build_transport_property_model(gas_model):
    viscosity_model = build_viscosity_model(gas_model)
    thermal_conductivity_model = build_thermal_conductivity_model(gas_model)
    return TransportPropertyModel(viscosity_model, thermal_conductivity_model)


class Config:
    _json_values = ["convective_flux", "viscous_flux", "solver", "grid",
                    "gas_model", "transport_properties"]
    __slots__ = _json_values

    def __init__(self):
        self.convective_flux = ConvectiveFlux()
        self.viscous_flux = ViscousFlux()
        self.solver = make_default_solver()
        self.gas_model = default_gas_model()
        self.transport_properties = build_transport_property_model(
            self.gas_model
        )

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


def main(file_name, res_dir):
    # make the config directory
    global DEFAULTS_DIRECTORY
    global SHARE_DIRECTORY
    DEFAULTS_DIRECTORY = f"{res_dir}/defaults"
    SHARE_DIRECTORY = res_dir
    directories = read_defaults(DEFAULTS_DIRECTORY,
                                "directories.json")
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
        "ViscousFlux": ViscousFlux,
        "Ausmdv": Ausmdv,
        "Hanel": Hanel,
        "Ldfss2": Ldfss2,
        "Block": Block,
        "Solver": Solver,
        "FlowState": FlowState,
        "GasState": GasState,
        "GasModel": GasModel,
        "IdealGas": IdealGas,
        "RungeKutta": RungeKutta,
        "supersonic_inflow": supersonic_inflow,
        "supersonic_outflow": supersonic_outflow,
        "slip_wall": slip_wall,
        "adiabatic_no_slip_wall": adiabatic_no_slip_wall,
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
