#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 22:39:23 2020

@author: root
"""
from fenics import *

mesh = Mesh("Mesh.xml")
Volume = MeshFunction("size_t", mesh, "Mesh_physical_region.xml")
bnd_mesh = MeshFunction("size_t", mesh, "Mesh_facet_region.xml")
#hdf = HDF5File(mesh.mpi_comm(), "Mesh.h5", "w")
#hdf.write(mesh, "/mesh")
#hdf.write(Volume, "/Volume")
#hdf.write(bnd_mesh, "/bnd_mesh")
#hdf.close()
xdmf = XDMFFile(mesh.mpi_comm(),"Mesh.xdmf")
xdmf.write(mesh)
xdmf.write(Volume)
xdmf = XDMFFile(mesh.mpi_comm(),"boundaries.xdmf")
xdmf.write(bnd_mesh)
xdmf.close()
