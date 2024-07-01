import pyvista as pv
import numpy as np
import analytic_solution
import pytest
import subprocess
import os
from parameters import angle, mach_1


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
    mach_2_analytic = analytic_solution.mach_2(mach_1, angle, 1.4)

    mesh = pv.read("plot/0001/block_0.vtu")
    mach_2_numerical = np.max(mesh.cell_data["Mach"])
    assert pytest.approx(mach_2_analytic, 0.01) == mach_2_numerical
