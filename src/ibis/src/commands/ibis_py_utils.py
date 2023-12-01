import json
from python_api import GasState, Vector3, PyIdealGas


def read_defaults(defaults_dir, file_name):
    with open(f"{defaults_dir}/{file_name}", "r") as defaults:
        defaults = json.load(defaults)
    return defaults

class FlowState:
    # __slots__ = ["p", "T", "rho" "vx", "vy", "vz"]
    __slots__ = ["gas", "vel"]

    def __init__(self, gas, vel=None, vx=None, vy=None, vz=None):
        self.gas = gas

        if vel and (vx or vy or vz):
            raise Exception("Velocity provided twice")

        if vel:
            self.vel = vel
        else:
            self.vel = Vector3()
            if vx:
                self.vel.x = vx
            if vy:
                self.vel.y = vy
            if vz:
                self.vel.z = vz



    def as_dict(self):
        return {
            "p": self.gas.p, "T": self.gas.T, 
            "rho": self.gas.rho, "energy": self.gas.energy,
            "vx": self.vel.x, "vy": self.vel.y, "vz": self.vel.z
        }

class GasModel:
    __slots__ = ["_gas_model"]

    def update_thermo_from_pT(self, gas_state):
        self._gas_model.update_thermo_from_pT(gas_state)

    def update_thermo_from_rhop(self, gas_state):
        self._gas_model.update_thermo_from_rhop(gas_state)

    def update_thermo_from_rhoT(self, gas_state):
        self._gas_model.update_thermo_from_rhoT(gas_state)

    def speed_of_sound(self, gas_state):
        return self._gas_model.speed_of_sound(gas_state)

    def validate(self):
        return

class IdealGas(GasModel):
    def __init__(self, args):
        if type(args) == float:
            self._gas_model = PyIdealGas(args)
        elif type(args) == dict:
            self._gas_model = PyIdealGas(args["R"])

    def as_dict(self):
        return {
            "type": "ideal_gas",
            "R": self._gas_model.R(),
            "Cv": self._gas_model.Cv(),
            "Cp": self._gas_model.Cp(),
            "gamma": self._gas_model.gamma()
        }


