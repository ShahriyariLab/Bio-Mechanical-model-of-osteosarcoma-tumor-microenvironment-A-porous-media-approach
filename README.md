# Bio-Mechanical-model-of-osteosarcoma-tumor-microenvironment-A-porous-media-approach

The code used to solve a Bio-Mechanical system of PDEs built for cancer modeling

An introduction to the files

/input contains data such as initial condition or parameter values

Main.py is the main module where all data is given or referred, all work in mathematics is presented, such as defining function spaces and finite elements, defining weak forms, calling solving commands and writting files.

GeoMaker.py is used to process the mesh generated from GMSH, a mesh generating software.

Mesh.h5 and Mesh.xdmf are generated from GMSH and the files for the mesh used in the problem

curvature.py is used to calculate the curvature of the boundary and guide shape change of the domain

jobUMASS3.qs is the commands for running the code on the UMass GHPCC

Main contributor: Navid Mirzaei
