import pyvista as pv
import numpy as np
import analytic_solution
import pytest
import subprocess
from parameters import angle, mach_1


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def run_simulation():
    command = "make all"
    subprocess.run(command.split())


def test_mach():
    run_simulation()
    mach_2_analytic = analytic_solution.mach_2(mach_1, angle, 1.4)

    mesh = pv.read("plot/0001/block_0.vtu")
    a = [0.74999, -0.10365, 0]
    cell_id = mesh.find_containing_cell(a)
    cell = mesh.extract_cells(cell_id)
    mach_2_numerical, = cell.cell_data["Mach"]
    assert pytest.approx(mach_2_analytic, 0.01) == mach_2_numerical
