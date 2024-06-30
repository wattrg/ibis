import pyvista as pv
import numpy as np
import analytic_solution
from parameters import angle, mach_1

np.bool = np.bool_

def test_mach():
    mach_2_analytic = analytic_solution.mach_2(mach_1, angle, 1.4)

    mesh = pv.read("plot/0010/block_0.vtu")
    a = [0.74999, -0.10365, 0]
    cell_id = mesh.find_containing_cell(a)
    cell = mesh.extract_cells(cell_id)
    mach_2_numerical, = cell.cell_data["Mach"]
    print(mach_2_analytic, mach_2_numerical)


test_mach()
