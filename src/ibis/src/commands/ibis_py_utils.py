import json
from python_api import GasState, Vector3


def read_defaults(defaults_dir, file_name):
    with open(f"{defaults_dir}/{file_name}", "r") as defaults:
        defaults = json.load(defaults)
    return defaults

class FlowState:
    # __slots__ = ["p", "T", "rho" "vx", "vy", "vz"]
    __slots__ = ["gas", "vel"]

    def __init__(self, p=None, T=None, rho=None, vx=0.0, vy=0.0, vz=0.0):
        self.vel = Vector3()
        self.vel.x = vx
        self.vel.y = vy
        self.vel.z = vz

        self.gas = GasState()
        if p and T:
            self.gas.p = p
            self.gas.T = T
            self.gas.rho = p / (287 * T)   

        elif p and rho:
            self.gas.p = p
            self.gas.rho = rho
            self.gas.T = p / (rho * 287)

        elif rho and T:
            self.gas.rho = rho
            self.gas.T = T
            self.gas.p = rho * 287.0 * T

        else:
            raise Exception("Need two of rho, T, and p")


    def as_dict(self):
        return {
            "p": self.gas.p, "T": self.gas.T, 
            "vx": self.vel.x, "vy": self.vel.y, "vz": self.vel.z
        }
