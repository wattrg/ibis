import gmsh

gmsh.initialize()

# general settings
gmsh.option.setNumber("General.Terminal", 1)
gmsh.option.setNumber("Mesh.Algorithm", 8) # quads
gmsh.option.setNumber("Mesh.RecombineAll", 1)
gmsh.option.setNumber("Mesh.SaveAll", 1)

# points
a = gmsh.model.geo.addPoint(0.0, 0.0, 0.0) # 1
b = gmsh.model.geo.addPoint(1.0, 0.0, 0.0) # 2
c = gmsh.model.geo.addPoint(1.0, 0.5, 0.0) # 3
d = gmsh.model.geo.addPoint(0.0, 0.5, 0.0) # 4

# lines
la = gmsh.model.geo.addLine(a, b)
lb = gmsh.model.geo.addLine(b, c)
lc = gmsh.model.geo.addLine(c, d)
ld = gmsh.model.geo.addLine(d, a)
cl = gmsh.model.geo.addCurveLoop([la, lb, lc, ld])

# surface
gmsh.model.geo.addPlaneSurface([cl])

# boundaries
gmsh.model.geo.addPhysicalGroup(1, [la], name="wall")
gmsh.model.geo.addPhysicalGroup(1, [ld], name="inflow")
gmsh.model.geo.addPhysicalGroup(1, [lb, lc], name="outflow")

# mesh size
gmsh.model.geo.mesh.setSize([(0, a), (0,b), (0,c), (0,d)], 0.009)

# syncronise the geometry so we can mesh
gmsh.model.geo.synchronize()

# generate the mesh
gmsh.model.mesh.generate(2)

# write the mesh
gmsh.write("grid.su2")

# open gmsh GUI to check the mesh
# gmsh.fltk.run()

gmsh.finalize()
