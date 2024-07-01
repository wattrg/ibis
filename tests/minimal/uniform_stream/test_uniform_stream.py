import pyvista as pv
import pytest
import subprocess
import os
import numpy as np


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def run_simulation():
    command = "make all"
    omp_vars = {"OMP_PLACES": "threads", "OMP_PROC_BIND": "spread"}
    env=dict(os.environ, **omp_vars)
    result = subprocess.run(command.split(), env=env)


def test_mach():
    run_simulation()

    mesh = pv.read("plot/0001/block_0.vtu")
    temperature = np.max(mesh.cell_data["temperature"])
    assert pytest.approx(temperature, 0.01) == 300
