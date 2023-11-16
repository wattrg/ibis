import gmsh

gmsh.initialize()

# general settings
gmsh.option.setNumber("General.Terminal", 1)
gmsh.option.setNumber("Mesh.Algorithm", 8) # quads
gmsh.option.setNumber("Mesh.RecombineAll", 1)
gmsh.option.setNumber("Mesh.SaveAll", 1)

# points
mesh_size = 0.0025
a = gmsh.model.geo.add_point(0.0, 0.646, 0.0, mesh_size)
b = gmsh.model.geo.add_point(0.975, 0.423, 0.0, mesh_size)
c = gmsh.model.geo.add_point(1.933, 0.081, 0.0, mesh_size)
d = gmsh.model.geo.add_point(2.3, 0.081, 0.0, mesh_size)
e = gmsh.model.geo.add_point(2.3, 0.0, 0.0, mesh_size)
f = gmsh.model.geo.add_point(0.0, 0.0, 0.0, mesh_size)

# lines
ramp_1 = gmsh.model.geo.add_line(a, b)
ramp_2 = gmsh.model.geo.add_line(b, c)
throat = gmsh.model.geo.add_line(c, d)
outlet = gmsh.model.geo.add_line(d, e)
symmetry = gmsh.model.geo.add_line(e, f)
inlet = gmsh.model.geo.add_line(f, a)
gmsh.model.geo.add_curve_loop([ramp_1, ramp_2, throat, outlet, symmetry, inlet])

# surface
gmsh.model.geo.add_plane_surface([1])

# boundaries
gmsh.model.geo.addPhysicalGroup(1, [ramp_1, ramp_2, symmetry, throat], name="wall")
gmsh.model.geo.addPhysicalGroup(1, [outlet], name="outflow")
gmsh.model.geo.addPhysicalGroup(1, [inlet], name="inflow") 

# syncronise the geometry so we can generate the mesh
gmsh.model.geo.synchronize()

# generate the mesh
gmsh.model.mesh.generate(2)

# write the mesh
gmsh.write("grid.su2")

# open gmsh GUI to check the mesh
# gmsh.fltk.run()

gmsh.finalize()
