R = 6.6e-03 -- sphere radies in meters

-- define points
a = Vector3:new{x=0.0, y=0.0}
b = Vector3:new{x=-R, y=0.0}
c = Vector3:new{x=0.0, y=R}
d = { Vector3:new{x=-2.0*R, y=0.0}, Vector3:new{x=-2.0*R, y=R},
      Vector3:new{x=-1.5*R, y=3*R}, Vector3:new{x=0.0, y=4.3*R} }

-- define paths
p1 = Line:new{p0=d[#d], p1=c}
p2 = Arc:new{p0=b, p1=c, centre=a}
p3 = Line:new{p0=d[1], p1=b}
p4 = Bezier:new{points=d}

-- define surface
psurf = makePatch{north=p1, east=p2, south=p3, west=p4
}

-- define grid clustering
-- cluster = RobertsFunction:new{end0=false,end1=true,beta=1.05}

-- define grid
sgrid = StructuredGrid:new{psurface=psurf,
                           cfList = {north=cluster, east=None,
                                     south=cluster, west=None},
                           niv=20, njv=20}
ugrid=UnstructuredGrid:new{sgrid=sgrid}
ugrid:write_to_su2_file("cylinder_eilmer.su2")


