import json
import os

IBIS = os.environ.get("IBIS")

def read_defaults(defaults_dir, file_name):
    with open(f"{defaults_dir}/{file_name}", "r") as defaults:
        defaults = json.load(defaults)
    return defaults

class FlowState:
    # __slots__ = ["p", "T", "rho" "vx", "vy", "vz"]

    def __init__(self, p=None, T=None, rho=None, vx=0.0, vy=0.0, vz=0.0):
        if p:
            self.p = p
        if T:
            self.T = T
        if rho:
            self.rho = rho
        self.vx = vx
        self.vy = vy
        self.vz = vz

        if not p:
            if not T or not rho:
                raise Exception("Need two of rho, T, and p")
            self.p = self.rho * 287 * T

        if not T:
            if not rho or not p:
                raise Exception("Need two of rho, T, and p")
            self.T = self.p / (self.rho * 287)

        if not rho:
            if not T or not p:
                raise Exception("Need two of rho, T, and p")
            self.rho = self.p / (287 * self.T)

    def as_dict(self):
        return {
            "p": self.p, "T": self.T, 
            "vx": self.vx, "vy": self.vy, "vz": self.vz
        }

