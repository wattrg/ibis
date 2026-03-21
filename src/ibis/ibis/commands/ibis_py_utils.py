import json
from python_api import Vector3
from pathlib import Path
from python_api import GridIO, vtk_type_from_elem_type
import numpy as np
import pyvista as pv
from typing import Self


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
            "p": self.gas.p,
            "T": self.gas.T,
            "rho": self.gas.rho,
            "energy": self.gas.energy,
            "vx": self.vel.x,
            "vy": self.vel.y,
            "vz": self.vel.z,
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
            "T_s": self._T_s,
        }


class ConstantPrandtlNumber:
    __slots__ = ["_Pr"]

    def __init__(self, Pr):
        self._Pr = Pr

    def as_dict(self):
        return {"type": "constant_prandtl_number", "Pr": self._Pr}


class TransportPropertyModel:
    __slots__ = ["_viscosity_model", "_thermal_conducitivty_model"]

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
            "thermal_conductivity": self._thermal_conducitivty_model.as_dict(),
        }

    def validate(self):
        return


class GasModel:
    __slots__ = ["_gas_model", "_transport_properties", "_type", "_species"]

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


def _grid_to_pyvista(grid: GridIO) -> pv.UnstructuredGrid:
    grid_vertices = grid.vertices()
    vertices = np.zeros((len(grid_vertices), 3))
    for i, grid_vertex in enumerate(grid_vertices):
        vertices[i, 0] = grid_vertex.pos().x
        vertices[i, 1] = grid_vertex.pos().y
        vertices[i, 2] = grid_vertex.pos().z

    grid_cells = grid.cells()
    cell_types = []
    cells = []
    for i, cell in enumerate(grid_cells):
        # fill out cell_types and cells
        cell_types.append(vtk_type_from_elem_type(cell.cell_type()))
        cell_vertices = cell.vertex_ids()
        cells.append(len(cell_vertices))
        for cell_vertex in cell_vertices:
            cells.append(cell_vertex)
    return pv.UnstructuredGrid(cells, cell_types, vertices)


def _read_flow_data(file, binary_format):
    if binary_format:
        return np.fromfile(file, dtype=np.float64)
    else:
        return np.loadtxt(file, dtype=np.float64)


class FlowSolution:
    def __init__(self, solution: pv.UnstructuredGrid):
        self._pv_mesh = solution
        self._cell_data_cache = self._pv_mesh.point_data_to_cell_data()

    @classmethod
    def from_grid(cls, grid: GridIO) -> Self:
        pv_mesh = _grid_to_pyvista(grid)
        return cls(pv_mesh)

    @classmethod
    def from_directory(cls, base_dir: Path | str, time_index: int = -1) -> Self:
        dir = Path(base_dir)

        # read simulation config
        with open(dir / "config" / "config.json") as f:
            config = json.load(f)

        if time_index == -1:
            with open(dir / "io" / "flow" / "flows", "r") as f:
                flow_dirs = f.readlines()
            flow_idxs = [int(flow_dir) for flow_dir in flow_dirs]
            time_index = flow_idxs[-1]

        # read the grid
        if config["grids"][0]["motion"]["enabled"]:
            grid_dir = dir / "io" / "grid" / f"{time_index:04}"
        else:
            grid_dir = dir / "io" / "grid" / "0000"
        flow_dir = dir / "io" / "flow" / f"{time_index:04}"
        grids = [
            GridIO(str(grid_dir / f"block_{i:04}.su2"), i)
            for i in range(len(config["grids"]))
        ]

        if config["io"]["flow_format"] == "native_binary":
            binary_flow_data = True
        else:
            binary_flow_data = False

        pv_meshs = []
        for grid in grids:
            # read the grid
            pv_mesh = _grid_to_pyvista(grid)

            # read corresponding flow data
            flow_dir_block = flow_dir / f"block_{grid.id():04}"
            pv_mesh.cell_data["pressure"] = _read_flow_data(
                flow_dir_block / "p", binary_flow_data
            )
            pv_mesh.cell_data["temperature"] = _read_flow_data(
                flow_dir_block / "T", binary_flow_data
            )
            pv_mesh.cell_data["vel"] = np.zeros((len(grid.cells()), 3))
            vx = _read_flow_data(flow_dir_block / "vx", binary_flow_data)
            vy = _read_flow_data(flow_dir_block / "vy", binary_flow_data)
            pv_mesh.cell_data["vel"][:, 0] = vx
            pv_mesh.cell_data["vel"][:, 1] = vy
            if grid.dim() == 3:
                vz = _read_flow_data(flow_dir_block / "vz", binary_flow_data)
                pv_mesh.cell_data["vel"][:, 2] = vz

            pv_meshs.append(pv_mesh)

        # join grids together
        flow_solution = pv.UnstructuredGrid()
        flow_solution = flow_solution.merge(pv_meshs)
        flow_solution.cell_data_to_point_data()
        return cls(flow_solution)

    def interpolate(self, other_solution: Self):
        self._pv_mesh = self._pv_mesh.sample(other_solution._pv_mesh)
        self._cell_data_cache = None

    def _to_cell_data(self):
        if self._cell_data_cache is None:
            self._cell_data_cache = self._pv_mesh.point_data_to_cell_data()
        return self._cell_data_cache

    def pressure(self) -> np.array:
        return self._to_cell_data().cell_data["pressure"]

    def temperature(self) -> np.array:
        return self._to_cell_data().cell_data["temperature"]

    def velocity(self) -> np.array:
        return self._to_cell_data().cell_data["vel"]
