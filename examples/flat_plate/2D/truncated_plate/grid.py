import gmsh

gmsh.initialize()

# general settings
gmsh.option.setNumber("General.Terminal", 1)
gmsh.option.setNumber("Mesh.Algorithm", 8) # quads
gmsh.option.setNumber("Mesh.RecombineAll", 1)
gmsh.option.setNumber("Mesh.SaveAll", 1)
gmsh.option.setNumber("Mesh.RecombineOptimizeTopology", 100)

# points
length = 0.5
height = 0.1

nx = 75
ny = 60
clustering = 0.95

a = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)
b = gmsh.model.geo.addPoint(length, 0.0, 0.0)
c = gmsh.model.geo.addPoint(length, height, 0.0)
d = gmsh.model.geo.addPoint(0.0, height, 0.0)

# lines
la = gmsh.model.geo.addLine(a, b)
lb = gmsh.model.geo.addLine(b, c)
lc = gmsh.model.geo.addLine(c, d)
ld = gmsh.model.geo.addLine(d, a)
cl = gmsh.model.geo.addCurveLoop([la, lb, lc, ld])

# surface
surface = gmsh.model.geo.addPlaneSurface([cl])

# transfinite entities
gmsh.model.geo.mesh.set_transfinite_curve(la, nx)
gmsh.model.geo.mesh.set_transfinite_curve(lc, nx)
gmsh.model.geo.mesh.set_transfinite_curve(ld, ny, "Progression",  clustering)
gmsh.model.geo.mesh.set_transfinite_curve(lb, ny, "Progression", -clustering)

gmsh.model.geo.mesh.set_transfinite_surface(surface)

# boundaries
gmsh.model.geo.addPhysicalGroup(1, [la], name="wall")
gmsh.model.geo.addPhysicalGroup(1, [ld], name="inflow")
gmsh.model.geo.addPhysicalGroup(1, [lb, lc], name="outflow")

# syncronise the geometry so we can mesh
gmsh.model.geo.synchronize()

gmsh.model.geo.mesh.set_recombine(2, 1)


gmsh.model.geo.synchronize()
gmsh.model.mesh.optimize('', niter=100)

# generate the mesh
gmsh.model.mesh.generate(2)

# write the mesh
gmsh.write("grid.su2")

# open gmsh GUI to check the mesh
# gmsh.fltk.run()

gmsh.finalize()
