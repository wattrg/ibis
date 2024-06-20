import gmsh
import math

gmsh.initialize()

# general settings
gmsh.option.setNumber("General.Terminal", 1)
gmsh.option.setNumber("Mesh.Algorithm", 8) # quads
gmsh.option.setNumber("Mesh.RecombineAll", 1)
gmsh.option.setNumber("Mesh.SaveAll", 1)

# points
inflow_length = 0.05
length = 0.9
angle = math.radians(15.0)
height = 0.6
dy = (length-inflow_length)*math.sin(angle)
a = gmsh.model.geo.addPoint(0.0, 0.0, 0.0) # 1 
b = gmsh.model.geo.addPoint(inflow_length, 0.0, 0.0) # 2
c = gmsh.model.geo.addPoint(length, -dy, 0.0) # 3
d = gmsh.model.geo.addPoint(length, height, 0.0) # 4
e = gmsh.model.geo.addPoint(inflow_length, height, 0.0)
f = gmsh.model.geo.addPoint(0.0, height, 0.0) # 5

# lines
ab = gmsh.model.geo.addLine(a, b)
bc = gmsh.model.geo.addLine(b, c)
cd = gmsh.model.geo.addLine(c, d)
de = gmsh.model.geo.addLine(d, e)
ef = gmsh.model.geo.addLine(e, f)
fa = gmsh.model.geo.addLine(f, a)

cl = gmsh.model.geo.addCurveLoop([ab, bc, cd, de, ef, fa])

# surface
surface = gmsh.model.geo.addPlaneSurface([cl])

# transfinite entities
cell_size = 0.005  # m
n_inflow = math.ceil(inflow_length / cell_size)
nx = math.ceil(length / cell_size)
ny = math.ceil(height / cell_size)
gmsh.model.geo.mesh.set_transfinite_curve(ab, n_inflow)
gmsh.model.geo.mesh.set_transfinite_curve(bc, nx)
gmsh.model.geo.mesh.set_transfinite_curve(cd, ny)
gmsh.model.geo.mesh.set_transfinite_curve(de, nx)
gmsh.model.geo.mesh.set_transfinite_curve(ef, n_inflow)
gmsh.model.geo.mesh.set_transfinite_curve(fa, ny)

gmsh.model.geo.mesh.set_transfinite_surface(surface, "Left", [a, c, d, f])

# boundaries
gmsh.model.geo.addPhysicalGroup(1, [ab, bc], name="wall")
gmsh.model.geo.addPhysicalGroup(1, [cd, de, ef], name="outflow")
gmsh.model.geo.addPhysicalGroup(1, [fa], name="inflow")


# syncronise the geometry so we can generate the mesh
gmsh.model.geo.synchronize()

# generate the mesh
gmsh.model.mesh.generate(2)

# write the mesh
gmsh.write("grid.su2")

# open gmsh GUI to check the mesh
# gmsh.fltk.run()

gmsh.finalize()
