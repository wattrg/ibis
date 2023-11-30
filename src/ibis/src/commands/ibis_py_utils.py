import json
from python_api import GasState, Vector3


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
            "vx": self.vel.x, "vy": self.vel.y, "vz": self.vel.z
        }
