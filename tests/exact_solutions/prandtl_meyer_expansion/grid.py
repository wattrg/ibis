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
e = gmsh.model.geo.addPoint(0.0, height, 0.0) # 5

# lines
la = gmsh.model.geo.addLine(1, 2) # 1
lb = gmsh.model.geo.addLine(2, 3) # 2
lc = gmsh.model.geo.addLine(3, 4) # 3
ld = gmsh.model.geo.addLine(4, 5) # 4
le = gmsh.model.geo.addLine(5, 1) # 5
cl = gmsh.model.geo.addCurveLoop([1,2,3,4,5])

# surface
surface = gmsh.model.geo.addPlaneSurface([1])

# transfinite entities
nx = 5
ny = 5
gmsh.model.geo.mesh.set_transfinite_curve(la, nx)
gmsh.model.geo.mesh.set_transfinite_curve(lb, nx)
gmsh.model.geo.mesh.set_transfinite_curve(lc, ny)
gmsh.model.geo.mesh.set_transfinite_curve(ld, 2*nx)
gmsh.model.geo.mesh.set_transfinite_curve(le, ny)

gmsh.model.geo.mesh.set_transfinite_surface(surface, "Left", [a, b, c, d, e])

# boundaries
gmsh.model.geo.addPhysicalGroup(1, [1,2], name="wall")
gmsh.model.geo.addPhysicalGroup(1, [3, 4], name="outflow")
gmsh.model.geo.addPhysicalGroup(1, [5], name="inflow") 

# mesh size
# gmsh.model.geo.mesh.setSize([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)], 0.009)

# syncronise the geometry so we can generate the mesh
gmsh.model.geo.synchronize()

# generate the mesh
gmsh.model.mesh.generate(2)

# write the mesh
gmsh.write("grid.su2")

# open gmsh GUI to check the mesh
# gmsh.fltk.run()

gmsh.finalize()
