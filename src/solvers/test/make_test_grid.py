"""
Generate a simple rectangular grid for testing graph colouring / Jacobian
assembly on an unstructured-style mesh (triangles, so connectivity isn't
as regular as a structured quad grid).

Domain: rectangle, length 2 x height 1.
Boundaries:
    - "symmetry" : bottom edge  (y = 0)
    - "wall"     : top edge     (y = height)
    - "inflow"   : left edge    (x = 0)
    - "outflow"  : right edge   (x = length)

Target: a couple hundred cells. Mesh size is tuned for that; adjust
`mesh_size` if you want more/fewer cells.

Usage:
    python make_test_grid.py
Produces:
    grid.su2   (SU2 format, for use with Ibis / your solver)
    grid.msh   (gmsh native format, handy for visual inspection in gmsh GUI)
"""

import gmsh

# ---------------------------------------------------------------- settings
length = 2.0
height = 1.0
mesh_size = 0.19  # tune this to change cell count (smaller = more cells)

gmsh.initialize()
gmsh.model.add("test_grid")

# ---------------------------------------------------------------- geometry
p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, mesh_size)
p2 = gmsh.model.geo.addPoint(length, 0.0, 0.0, mesh_size)
p3 = gmsh.model.geo.addPoint(length, height, 0.0, mesh_size)
p4 = gmsh.model.geo.addPoint(0.0, height, 0.0, mesh_size)

l_bottom = gmsh.model.geo.addLine(p1, p2)  # symmetry
l_outflow = gmsh.model.geo.addLine(p2, p3)  # outflow
l_top = gmsh.model.geo.addLine(p3, p4)  # wall
l_inflow = gmsh.model.geo.addLine(p4, p1)  # inflow

loop = gmsh.model.geo.addCurveLoop([l_bottom, l_outflow, l_top, l_inflow])
surface = gmsh.model.geo.addPlaneSurface([loop])

gmsh.model.geo.synchronize()

# ---------------------------------------------------------------- physical groups
# Boundaries
symmetry_group = gmsh.model.addPhysicalGroup(1, [l_bottom])
gmsh.model.setPhysicalName(1, symmetry_group, "symmetry")

wall_group = gmsh.model.addPhysicalGroup(1, [l_top])
gmsh.model.setPhysicalName(1, wall_group, "wall")

inflow_group = gmsh.model.addPhysicalGroup(1, [l_inflow])
gmsh.model.setPhysicalName(1, inflow_group, "inflow")

outflow_group = gmsh.model.addPhysicalGroup(1, [l_outflow])
gmsh.model.setPhysicalName(1, outflow_group, "outflow")

# Domain (cells)
fluid_group = gmsh.model.addPhysicalGroup(2, [surface])
gmsh.model.setPhysicalName(2, fluid_group, "fluid")

# ---------------------------------------------------------------- mesh
# Triangular mesh (unstructured connectivity - good stress test for
# colouring, since cell degree/adjacency is irregular compared to a
# structured quad grid).
gmsh.model.mesh.generate(2)

num_elements = len(gmsh.model.mesh.getElements(2)[1][0])
print(f"Generated mesh with approximately {num_elements} triangular cells")
gmsh.fltk.run()
# ---------------------------------------------------------------- write output
gmsh.write("grid.su2")

gmsh.finalize()
