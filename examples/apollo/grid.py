import gmsh
import math

gmsh.initialize()

gmsh.option.set_number("General.Terminal", 1)
gmsh.option.set_number("Mesh.Algorithm", 8)
gmsh.option.set_number("Mesh.RecombineAll", 1)
gmsh.option.set_number("Mesh.SaveAll", 1)
# gmsh.option.set_number("Mesh.MeshSizeFromCurvature", 10)
# gmsh.option.set_number("Mesh.MeshSizeFromPoints", 0)

size = 0.025
R = 10.0
H = 0.75
offset = 7

# capsule points
a = gmsh.model.geo.add_point(-0.02058233255, 0.0, 0.0, size)
p1 = gmsh.model.geo.add_point(0.3743, 1.8368, 0.0, size)
p2 = gmsh.model.geo.add_point(0.5543, 1.9558, 0.0, size)
p3 = gmsh.model.geo.add_point(0.6608, 1.9242, 0.0, size)
p4 = gmsh.model.geo.add_point(3.3254, 0.1938, 0.0, size)
p5 = gmsh.model.geo.add_point(3.4306, 0.0, 0.0, size)
c = gmsh.model.geo.add_point(4.448807667, 0.0, 0.0)
d = gmsh.model.geo.add_point(0.554227, 1.760289, 0.0)
e = gmsh.model.geo.add_point(3.1995, 0.0, 0.0)

# capsule curves
arc1 = gmsh.model.geo.add_circle_arc(a, c, p1)
arc2 = gmsh.model.geo.add_circle_arc(p1, d, p3)
l1 = gmsh.model.geo.add_line(p3, p4)
arc3 = gmsh.model.geo.add_circle_arc(p4, e, p5)

# far-field points
f1 = gmsh.model.geo.add_point(R+offset, 0.0, 0.0, H)
f2 = gmsh.model.geo.add_point(offset, R, 0, H)
f3 = gmsh.model.geo.add_point(-R+offset, 0.0, 0.0, H)
fc = gmsh.model.geo.add_point(offset, 0.0, 0.0, H)

# far-field lines
ff1 = gmsh.model.geo.add_circle_arc(f1, fc, f2)
ff2 = gmsh.model.geo.add_circle_arc(f2, fc, f3)
ff_symmetry_forebody = gmsh.model.geo.add_line(f3, a)
ff_symmetry_aftbody = gmsh.model.geo.add_line(p5, f1)

# surfaces
cl = gmsh.model.geo.add_curve_loop(
    [arc1, arc2, l1, arc3, ff_symmetry_aftbody, ff1, ff2, ff_symmetry_forebody]
)
ps = gmsh.model.geo.add_plane_surface([1])

# gmsh's built in CAD engine can't revolve pi radians or more,
# so we have to break the revolution up into 3 parts
ov = gmsh.model.geo.revolve([(2, ps)], 0, 0, 0, 1, 0, 0, 2 * math.pi / 3, [21], recombine=True)
ov = gmsh.model.geo.revolve([(2, 40)], 0, 0, 0, 1, 0, 0, 2 * math.pi / 3, [21], recombine=True)
ov = gmsh.model.geo.revolve([(2, 72)], 0, 0, 0, 1, 0, 0, 2 * math.pi / 3, [21], recombine=True)

gmsh.model.geo.add_physical_group(
    2, [20, 24, 28, 31, 52, 56, 60, 63, 84, 88, 92, 95], name="capsule"
)
gmsh.model.geo.add_physical_group(
    2, [35, 67, 99], name="outflow"
)
gmsh.model.geo.add_physical_group(
    2, [38, 70, 102], name="inflow"
)

gmsh.model.geo.synchronize()

gmsh.model.mesh.generate(3)

gmsh.write("grid.su2")
# gmsh.fltk.run()

gmsh.finalize()
