import json
from python_api import Vector3


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


class SutherlandViscosity:
    __slots__ = ["_mu_0", "_T_0", "_T_s"]

    def __init__(self, mu_0, T_0, T_s):
        self._mu_0 = mu_0
        self._T_0 = T_0
        self._T_s = T_s

    def as_dict(self):
        return {
            "type": "sutherland",
            "mu_0": self._mu_0,
            "T_0": self._T_0,
            "T_s": self._T_s
        }


class ConstantPrandtlNumber:
    __slots__ = ["_Pr"]

    def __init__(self, Pr):
        self._Pr = Pr

    def as_dict(self):
        return {
            "type": "constant_prandtl_number",
            "Pr": self._Pr
        }


class TransportPropertyModel:
    __slots__ = ["_viscosity_model",
                 "_thermal_conducitivty_model"]

    def __init__(self, viscosity, thermal_conductivity):
        self._viscosity_model = viscosity
        self._thermal_conducitivty_model = thermal_conductivity

    def viscosity_model(self):
        return self._viscosity_model

    def thermal_conductivity_model(self):
        return self._thermal_conducitivty_model

    def as_dict(self):
        return {
            "viscosity": self._viscosity_model.as_dict(),
            "thermal_conductivity": self._thermal_conducitivty_model.as_dict()
        }

    def validate(self):
        return


class GasModel:
    __slots__ = ["_gas_model", "_transport_properties", "_type",
                 "_species"]

    def update_thermo_from_pT(self, gas_state):
        self._gas_model.update_thermo_from_pT(gas_state)

    def update_thermo_from_rhop(self, gas_state):
        self._gas_model.update_thermo_from_rhop(gas_state)

    def update_thermo_from_rhoT(self, gas_state):
        self._gas_model.update_thermo_from_rhoT(gas_state)

    def speed_of_sound(self, gas_state):
        return self._gas_model.speed_of_sound(gas_state)

    def species(self):
        return self._species

    def validate(self):
        return

    def type(self):
        return self._type

    def as_dict(self):
        return self._gas_model.as_dict()
