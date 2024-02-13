import gmsh

gmsh.initialize()

# general settings
gmsh.option.setNumber("General.Terminal", 1)
gmsh.option.setNumber("Mesh.Algorithm", 8) # quads
gmsh.option.setNumber("Mesh.RecombineAll", 1)
gmsh.option.setNumber("Mesh.SaveAll", 1)

# points
length = 1.0
height = 0.5
jet_width = 0.01
a = gmsh.model.geo.addPoint(0.0, 0.0, 0.0) 
b = gmsh.model.geo.addPoint(length, 0.0, 0.0)
c = gmsh.model.geo.addPoint(length, height, 0.0)
d = gmsh.model.geo.addPoint(0.0, height, 0.0)
e = gmsh.model.geo.addPoint(0.0, height/2+jet_width/2, 0.0)
f = gmsh.model.geo.addPoint(0.0, height/2-jet_width/2, 0.0)

# lines
la = gmsh.model.geo.addLine(a, b)
lb = gmsh.model.geo.addLine(b, c)
lc = gmsh.model.geo.addLine(c, d)
ld = gmsh.model.geo.addLine(d, e)
le = gmsh.model.geo.addLine(e, f)
lf = gmsh.model.geo.addLine(f, a)
cl = gmsh.model.geo.addCurveLoop([la, lb, lc, ld, le, lf])

# surface
gmsh.model.geo.addPlaneSurface([cl])

# boundaries
gmsh.model.geo.addPhysicalGroup(1, [ld, lf], name="wall")
gmsh.model.geo.addPhysicalGroup(1, [le], name="inflow")
gmsh.model.geo.addPhysicalGroup(1, [lb, la, lc], name="outflow")

# mesh size
size = length / 500
gmsh.model.geo.mesh.setSize([(0, a), (0,b), (0,c), (0,d)], size)
gmsh.model.geo.mesh.setSize([(0, e), (0,f)], size/2)

# syncronise the geometry so we can mesh
gmsh.model.geo.synchronize()

gmsh.model.geo.synchronize()

# generate the mesh
gmsh.model.mesh.generate(2)

# write the mesh
gmsh.write("grid.su2")

# open gmsh GUI to check the mesh
# gmsh.fltk.run()

gmsh.finalize()
