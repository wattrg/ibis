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

cell_size = 0.01
gmsh.model.geo.addPoint(0.0, 0.0, 0.0, cell_size) # 1 
gmsh.model.geo.addPoint(inflow_length, 0.0, 0.0, cell_size) # 2
gmsh.model.geo.addPoint(length, -dy, 0.0, cell_size) # 3
gmsh.model.geo.addPoint(length, height, 0.0, cell_size) # 4
gmsh.model.geo.addPoint(0.0, height, 0.0, cell_size) # 5

# lines
gmsh.model.geo.addLine(1, 2) # 1
gmsh.model.geo.addLine(2, 3) # 2
gmsh.model.geo.addLine(3, 4) # 3
gmsh.model.geo.addLine(4, 5) # 4
gmsh.model.geo.addLine(5, 1) # 5
gmsh.model.geo.addCurveLoop([1,2,3,4,5])

# surface
gmsh.model.geo.addPlaneSurface([1])

# boundaries
gmsh.model.geo.addPhysicalGroup(1, [1, 2], name="wall")
gmsh.model.geo.addPhysicalGroup(1, [3, 4], name="outflow")
gmsh.model.geo.addPhysicalGroup(1, [5], name="inflow")

# syncronise the geometry so we can generate the mesh
gmsh.model.geo.synchronize()

# generate the mesh
gmsh.model.mesh.generate(2)

# write the mesh
gmsh.write("grid.su2")

# open gmsh GUI to check the mesh
# gmsh.fltk.run()

gmsh.finalize()
