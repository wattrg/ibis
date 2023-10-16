import gmsh

gmsh.initialize()

# general settings
gmsh.option.setNumber("General.Terminal", 1)
gmsh.option.setNumber("Mesh.Algorithm", 8) # quads
gmsh.option.setNumber("Mesh.RecombineAll", 1)
gmsh.option.setNumber("Mesh.SaveAll", 1)

# points
gmsh.model.geo.addPoint(0.0, 0.0, 0.0) # 1
gmsh.model.geo.addPoint(0.6, 0.0, 0.0) # 2
gmsh.model.geo.addPoint(0.6, 0.2, 0.0) # 3
gmsh.model.geo.addPoint(3.0, 0.2, 0.0) # 4
gmsh.model.geo.addPoint(3.0, 1.0, 0.0) # 5
gmsh.model.geo.addPoint(0.0, 1.0, 0.0) # 6

# lines
gmsh.model.geo.addLine(1, 2)
gmsh.model.geo.addLine(2, 3)
gmsh.model.geo.addLine(3, 4)
gmsh.model.geo.addLine(4, 5)
gmsh.model.geo.addLine(5, 6)
gmsh.model.geo.addLine(6, 1)
gmsh.model.geo.addCurveLoop([1,2,3,4,5,6])

# surface
gmsh.model.geo.addPlaneSurface([1])

# boundaries
gmsh.model.addPhysicalGroup(1, [1,2,3,5], name="wall")
gmsh.model.addPhysicalGroup(1, [6], name="inflow")
gmsh.model.addPhysicalGroup(1, [4], name="outflow")

# mesh size
gmsh.model.geo.mesh.setSize([(0, 1), (0,2), (0,3), (0,4), (0,5), (0,6)], 0.009)

# syncronise the geometry so we can mesh
gmsh.model.geo.synchronize()

# generate the mesh
gmsh.model.mesh.generate(2)

# write the mesh
gmsh.write("grid.su2")

# open gmsh GUI to check the mesh
# gmsh.fltk.run()

gmsh.finalize()
