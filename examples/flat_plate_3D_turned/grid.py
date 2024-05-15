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
height = 0.25
width = 1.0

niv = 50
njv = 50
nkv = 100

clustering = 0.9

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

# volume
volume = gmsh.model.geo.extrude([(2, surface)], 0, 0, width)

# transfinite curves
gmsh.model.geo.mesh.set_transfinite_curve(4, njv, "Progression", clustering)
gmsh.model.geo.mesh.set_transfinite_curve(9, njv, "Progression", clustering)
gmsh.model.geo.mesh.set_transfinite_curve(7, njv, "Progression", -clustering)
gmsh.model.geo.mesh.set_transfinite_curve(2, njv, "Progression", -clustering)
gmsh.model.geo.mesh.set_transfinite_curve(8, niv)
gmsh.model.geo.mesh.set_transfinite_curve(3, niv)
gmsh.model.geo.mesh.set_transfinite_curve(1, niv)
gmsh.model.geo.mesh.set_transfinite_curve(6, niv)
gmsh.model.geo.mesh.set_transfinite_curve(20, nkv)
gmsh.model.geo.mesh.set_transfinite_curve(11, nkv)
gmsh.model.geo.mesh.set_transfinite_curve(16, nkv)
gmsh.model.geo.mesh.set_transfinite_curve(12, nkv)

# transfinite surface
tfs = gmsh.model.geo.mesh.set_transfinite_surface(1, "Left")
tfs = gmsh.model.geo.mesh.set_transfinite_surface(13, "Left")
tfs = gmsh.model.geo.mesh.set_transfinite_surface(17, "left")
tfs = gmsh.model.geo.mesh.set_transfinite_surface(21, "Left")
tfs = gmsh.model.geo.mesh.set_transfinite_surface(25, "Left")
tfs = gmsh.model.geo.mesh.set_transfinite_surface(26, "Left")

# transfitinite volume
tfv = gmsh.model.geo.mesh.set_transfinite_volume(1, [1, 2, 3, 4, 5, 6, 10, 14])

# boundaries
gmsh.model.geo.addPhysicalGroup(2, [17, 25], name="sides")
gmsh.model.geo.addPhysicalGroup(2, [13], name="wall")
gmsh.model.geo.addPhysicalGroup(2, [21], name="top")
gmsh.model.geo.addPhysicalGroup(2, [26], name="inflow")
gmsh.model.geo.addPhysicalGroup(2, [1], name="outflow")

gmsh.model.geo.mesh.set_recombine(2, 1)

gmsh.model.geo.synchronize()

# generate the mesh
gmsh.model.mesh.generate(3)

# write the mesh
gmsh.write("grid.su2")

# open gmsh GUI to check the mesh
# gmsh.fltk.run()

gmsh.finalize()
