import gmsh

gmsh.initialize()

# general settings
gmsh.option.setNumber("General.Terminal", 1)
# gmsh.option.setNumber("Mesh.Algorithm", 8) # delauny
# gmsh.option.setNumber("Mesh.RecombineAll", 1)
gmsh.option.setNumber("Mesh.SaveAll", 1)

# geometric properties
size = 2e-4
R = 6.6e-03  # sphere radius in meters

# define points
centre = gmsh.model.geo.add_point(0.0, 0.0, 0.0, size)
a = gmsh.model.geo.add_point(-R, 0.0, 0.0, size)
b = gmsh.model.geo.add_point(0.0, R, 0.0, size)
c = []
c.append(gmsh.model.geo.add_point(-2.0*R, 0.0, 0.0, size))
c.append(gmsh.model.geo.add_point(-2.0*R, 1.5*R,   0.0, size))
c.append(gmsh.model.geo.add_point(-1.5*R,     3*R, 0.0, size))
c.append(gmsh.model.geo.add_point(0.0,    4*R, 0.0, size))

shock_pts = []
shock_pts.append(gmsh.model.geo.add_point(-1.8*R, 0.0, 0.0, size))
shock_pts.append(gmsh.model.geo.add_point(-1.8*R, 1.3*R, 0.0, size))
shock_pts.append(gmsh.model.geo.add_point(-1.3*R, 2.8*R, 0.0, size))
shock_pts.append(gmsh.model.geo.add_point(0.0*R, 3.8*R, 0.0, size))

# define paths
freestream_outflow = gmsh.model.geo.add_line(shock_pts[-1], c[-1])
outflow = gmsh.model.geo.add_line(shock_pts[-1], b)
wall = gmsh.model.geo.add_circle_arc(b, centre, a)
symmetry = gmsh.model.geo.add_line(a, shock_pts[0])
freestream_symmetry = gmsh.model.geo.add_line(c[0], shock_pts[0])
inflow = gmsh.model.geo.add_bezier(c[::-1])
shock = gmsh.model.geo.add_bezier(shock_pts)
shock_reverse = gmsh.model.geo.add_bezier(shock_pts[::-1])

# define curve loop
shock_layer = gmsh.model.geo.add_curve_loop([outflow, wall, symmetry, shock])
freestream_layer = gmsh.model.geo.add_curve_loop(
    [freestream_outflow, shock, freestream_symmetry, inflow]
)

# define surface
gmsh.model.geo.add_plane_surface([shock_layer])
gmsh.model.geo.add_plane_surface([freestream_layer])

# define boundary names
gmsh.model.geo.add_physical_group(1, [outflow, freestream_outflow], name="outflow")
gmsh.model.geo.add_physical_group(1, [wall], name="wall")
gmsh.model.geo.add_physical_group(1, [symmetry, freestream_symmetry], name="symmetry")
gmsh.model.geo.add_physical_group(1, [inflow], name="inflow")
gmsh.model.geo.add_physical_group(1, [shock], name="shock")

gmsh.model.geo.synchronize()

# define prism layer
# boundary_layer = gmsh.model.mesh.field.add("BoundaryLayer")
# gmsh.model.mesh.field.set_numbers(boundary_layer, "CurvesList", [2])
# gmsh.model.mesh.field.set_numbers(boundary_layer, "PointsList", [a,b])
# gmsh.model.mesh.field.set_number(boundary_layer, "SizeFar", size)
# gmsh.model.mesh.field.set_number(boundary_layer, "Size", 1e-6)
# gmsh.model.mesh.field.set_number(boundary_layer, "Quads", 1)
# gmsh.model.mesh.field.set_number(boundary_layer, "Ratio", 1.1)
# gmsh.model.mesh.field.set_number(boundary_layer, "Thickness", 3.0e-4)
# gmsh.model.mesh.field.setAsBoundaryLayer(boundary_layer)

# generate 2D grid
gmsh.model.mesh.generate(2)

# write grid to SU2 format
gmsh.write("cylinder_gmsh.su2")

# uncomment the line below to visualize the mesh
# gmsh.fltk.run()

gmsh.finalize()
