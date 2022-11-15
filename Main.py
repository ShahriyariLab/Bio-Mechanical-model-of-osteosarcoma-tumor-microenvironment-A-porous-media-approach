from dolfin import *
import logging
import scipy.optimize as op
import numpy as np
import pandas as pd
from subprocess import call
from GeoMaker import *
from curvature import *
# from IC_Loc_DG import *
from ufl import Min
# from smoothen import *
from ufl import nabla_div
import matplotlib.pyplot as plt
###############################################################
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}
###############################################################

#Remeshing info
#These parameters will be fed to GeoMaker for remeshing
# refine = (No refine=1, Other integers bigger than 1 for refinement)')
# Org_size = Original Mesh size
# Max_cellnum = Max number of cells desired after  remeshing?
# Min_cellnum = Min number of cells desired after  remeshing?
refine = 1
Refine = str(refine)
Org_size = 0.0026
Max_cellnum = 3400
Min_cellnum = 3300
remesh_step = 60000 #remeshing not used here, saving for future uses
###############################################################

#Reporting options
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("rothemain.rothe_utils")
logging.getLogger('UFL').setLevel(logging.WARNING)
logging.getLogger('FFC').setLevel(logging.WARNING)
###############################################################

#MPI parameters for parallelizing. (Not applicable for this study)
comm = MPI.comm_world
num_processes=  comm.Get_size()
this_process = comm.Get_rank()
###############################################################

#Solvers compatible with Parallelizing (Not applicable for this study)
# Test for PETSc or Tpetra
if not has_linear_algebra_backend("PETSc") and not has_linear_algebra_backend("Tpetra"):
    info("DOLFIN has not been configured with Trilinos or PETSc. Exiting.")
    exit()

if not has_krylov_solver_preconditioner("amg"):
    info("Sorry, this demo is only available when DOLFIN is compiled with AMG "
         "preconditioner, Hypre or ML.")
    exit()

if has_krylov_solver_method("minres"):
    krylov_method = "minres"
elif has_krylov_solver_method("tfqmr"):
    krylov_method = "tfqmr"
else:
    info("Default linear algebra backend was not compiled with MINRES or TFQMR "
         "Krylov subspace method. Terminating.")
    exit()
###############################################################

#Solver configs (Not applicable for this study)
class Problem(NonlinearProblem):
    def __init__(self, J, F, bcs):
        self.bilinear_form = J
        self.linear_form = F
        self.bcs = bcs
        NonlinearProblem.__init__(self)

    def F(self, b, x):
        assemble(self.linear_form, tensor=b)
        for bc in self.bcs:
            bc.apply(b, x)

    def J(self, A, x):
        assemble(self.bilinear_form, tensor=A)
        for bc in self.bcs:
            bc.apply(A)
###############################################################

class CustomSolver1(NewtonSolver):
    def __init__(self):
        NewtonSolver.__init__(self, mesh.mpi_comm(),
                              PETScKrylovSolver(), PETScFactory.instance())

    def solver_setup(self, A, P, problem, iteration):
        self.linear_solver().set_operator(A)
        PETScOptions.set("ksp_reuse_preconditioner","true")
        PETScOptions.set("ksp_max_it", 200)
        PETScOptions.set("ksp_initial_guess_nonzero", "true")
        PETScOptions.set("ksp_type", "gmres")
        PETScOptions.set("ksp_monitor")
        PETScOptions.set("pc_type", "hypre")
        PETScOptions.set("pc_hypre_type", "euclid")

        self.linear_solver().set_from_options()
###############################################################
###############################################################80



# Nullspace of rigid motions defined as Lagrange multiplier in the weak mode for the mechanical problem
## Translation 3D
#Z_transl = [Constant((1, 0, 0)), Constant((0, 1, 0)), Constant((0, 0, 1))]
##Translation 2D
Z_transl = [Constant((1, 0)), Constant((0, 1))]
# Rotations 3D
#Z_rot = [Expression(('0', 'x[2]', '-x[1]')),
#         Expression(('-x[2]', '0', 'x[0]')),
#         Expression(('x[1]', '-x[0]', '0'))]
# Rotations 2D
Z_rot = [Expression(('-x[1]', 'x[0]'),degree=0)]
# All
Z = Z_transl + Z_rot
###############################################################
###############################################################


#Load coarse mesh
#Parallel compatible Mesh readings
mesh= Mesh()
xdmf = XDMFFile(mesh.mpi_comm(), "Mesh.xdmf")
xdmf.read(mesh)
mvc = MeshValueCollection("size_t", mesh, 2)
with XDMFFile("Mesh.xdmf") as infile:
    infile.read(mvc, "f")
Volume = cpp.mesh.MeshFunctionSizet(mesh, mvc)
xdmf.close()
mvc2 = MeshValueCollection("size_t", mesh, 1)
with XDMFFile("boundaries.xdmf") as infile:
    infile.read(mvc2, "f")
bnd_mesh = cpp.mesh.MeshFunctionSizet(mesh, mvc2)
###############################################################

#Saving the mesh files
File('Results/mesh.pvd')<<mesh
#File('Results/Volume.pvd')<<Volume
#File('Results/boundary.pvd')<<bnd_mesh
###############################################################

# Build function space
P22 = VectorElement("P", mesh.ufl_cell(), 4)
P00 = VectorElement("R", mesh.ufl_cell(), 0, dim=3)
Q = FiniteElement("P", mesh.ufl_cell(), 1)
element = MixedElement([P22, P00, Q])
W = FunctionSpace(mesh, element)

S1 = FunctionSpace(mesh,'P',1)
VV = VectorFunctionSpace(mesh,'Lagrange',4)
VV1 = VectorFunctionSpace(mesh,'Lagrange',1)
R = FunctionSpace(mesh,'R',0)
P1 = FiniteElement('P', triangle,1)
P3 = FiniteElement('P', triangle,1)
PB = FiniteElement('B', triangle,3)
NEE = NodalEnrichedElement(P1, PB)   #Bubble stabilization for th biological system
element1 = MixedElement([NEE,NEE,NEE,NEE,NEE,NEE,NEE,NEE,NEE,NEE,NEE,NEE,NEE,NEE])
Mixed_Space1 = FunctionSpace(mesh, element1)
###############################################################


#Defining functions and test functions
U = Function(Mixed_Space1)
U_n = Function(Mixed_Space1)

Mn, M, Tn, Th, Tr, Tc, Dn, D, C, N, Ig, mu1, mu2, H = split(U)
Mn_n, M_n, Tn_n, Th_n, Tr_n, Tc_n, Dn_n, D_n, C_n, N_n, Ig_n, mu1_n, mu2_n, H_n = split(U_n)
v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14 = TestFunctions(Mixed_Space1)

## 2 ##
bcu = DirichletBC(W.sub(0), Constant((0.0, 0.0)), DomainBoundary())
#p_bc = Expression('x[0]*x[1]', degree = 1)
bcp = DirichletBC(W.sub(2), Constant(-13000.0), DomainBoundary())
bc = [bcu, bcp]

# Define initial value
bfU_n = Function(W)
bfu_n, l_n, p_n = split(bfU_n)

bfu, l, p = TrialFunctions(W)
bfv, w, q = TestFunctions(W)

#######################################################################

# Construct integration measure using these markers
ds = Measure('ds', subdomain_data=bnd_mesh)
dx = Measure('dx', subdomain_data=Volume)
###############################################################

# reading mean-based cell data for sensitivity analysis
max_values = pd.read_csv('input/input/max_values.csv')
max_values = max_values.values
parameters = pd.read_csv('input/input/parameters11.csv').values
###############################################################

Pars =['\lambda_{\mu_1T_h}','\lambda_{\mu_1M}','\lambda_{\mu_1C}',
                        '\lambda_{\mu_2T_h}','\lambda_{\mu_2M}',
                        '\lambda_{\mu_2C}',
                        '\lambda_{I_\gamma T_h}','\lambda_{I_\gamma T_c}',
                        '\lambda_{HM}','\lambda_{HD}','\lambda_{HN}',
                        '\lambda_{MI_\gamma}','\lambda_{M\mu_1}','\lambda_{MI_\gamma}',
                        '\lambda_{M\mu_1}',
                        '\lambda_{T_hM}','\lambda_{T_hD}','\lambda_{T_r\mu_1}','\lambda_{T_cT_h}','\lambda_{T_cM}',
                        '\lambda_{T_cD}','\lambda_{T_hM}','\lambda_{T_hD}','\lambda_{T_r\mu_1}',
                        '\lambda_{T_cT_h}','\lambda_{T_cM}','\lambda_{T_cD}','\lambda_{DC}','\lambda_{DH}',
                        '\lambda_{DC}','\lambda_{DH}', '\lambda_{C}', '\lambda_{C\mu_1}', '\lambda_{C\mu_2}',
                        '\delta_{\mu_1}','\delta_{\mu_2}','\delta_{I_\gamma}',
                        '\delta_{H}',
                        '\delta_{M}',
                        '\delta_{M_N}','\delta_{T_hT_r}',
                        '\delta_{T_h\mu_1}',
                        '\delta_{T_h}',
                        '\delta_{M}',
                        '\delta_{T_cT_r}','\delta_{T_c\mu_1}',
                        '\delta_{T_c}',
                        '\delta_{T_N}',
                        '\delta_{DC}',
                        '\delta_D','\delta_{D_N}','\delta_{CT_c}',
                        '\delta_{CI_\gamma}', '\delta_C'
                        '\delta_{CT_c}', '\delta_{CI_\gamma}', '\delta_C', '\delta_{N}'
                        'A_{M_N}','A_{T_N}','A_{D_N}',
                        '\alpha_{NC}','C_0']
###############################################################

#Reading parameters
par = parameters[0]
###############################################################


#PDE Parameters dimensional
#Reference for D_cell: Serum uPAR as Biomarker in Breast Cancer Recurrence: A Mathematical Model
#Reference for D_cyto: The role of CD200–CD200R in tumorimmuneevasion
#Reference for D_H: Mathematical model on Alzheimer’s disease
#They were all in CM2/day so I had to change it to mm2 by myltiplying by 100 and /hour by dividing by 24. the following is in mm2/hours
D_cell, D_H, D_cyto =  3.6e-8, 3.3e-3, 5.2e-5
kappaTh, kappaTc, kappaTr, kappaDn, kappaD, kappaM = 1, 1, 1, 1, 1, 1
coeff = Constant(1)    #advection constant
maxmu1 = Constant(26.86761)
maxmu2 = Constant(3.733365)
maxIg = Constant(2.078405)
maxH = Constant(5.7194)
maxM = Constant(21967004)
maxMn = Constant(25513759)
maxTh = Constant(11025134)
maxTr = Constant(5031409)
maxTc = Constant(31680961)
maxTn = Constant(11956654)
maxD = Constant(1972965)
maxDn = Constant(820243)
maxC = Constant(13900000000)
maxN = Constant(942000000)
TotalD = Constant(76412021)

# list of parameter values
K0 = Constant(40000)
G0 = Constant(30000)
c0 = Constant(5.0e-6)
alpha0 = Constant(0.7)
kappa0 = Constant(6.9e-14)
dyViscBlood = Constant(3.5e-3)
phi = Constant(0.2)
cellVolume = 0.8*Constant(pi*10e-10)
DarcyConst = kappa0

#time step variables
T = 2500    # final
num_steps= 4002 #number of time steps
dt = 0.25 # time step
eps = 1       # diffusion coefficient
t=0
k = Constant(dt)
###############################################################

# different initial conditions
U_0 = Expression(('2.367', '1.005', '0.019', '0.794', '0.764', '0.828', '1.122', '0', '0.020', '0.160', \
'2.394', '1.104', '1.806', '1.059'), degree = 2)
# More M and Tc initially
# U_0 = Expression(('2.367', '1.005 + 10*exp(-5e5*x[0]*x[0] - 5e5*x[1]*x[1])', '0.019', '0.794', '0.764', '0.828 + 8*exp(-5e5*x[0]*x[0] - 5e5*x[1]*x[1])', '1.122', '0', '0.020', '0.160', \
# '2.394', '1.104', '1.806', '1.059'), degree = 2)
# other clusters
#U_0 = Expression(('1.307', '3.259', '0.859', '0.988', '0.753', '0.954', '1.451', '2.313', '0.062', '1.299', \
#'0.693', '0.071', '0.005', '0.018'), degree = 2)
U_n = project(U_0, Mixed_Space1)

#  0 ,   1 ,   2  ,  3 , 4  , 5   ,  6  ,  7  ,  8  ,  9  , 10 , 11  , 12 , 13
Mn_n, M_n, Tn_n, Th_n, Tr_n, Tc_n, Dn_n, D_n, C_n, N_n, Ig_n, mu1_n, mu2_n, H_n = U_n.split()
##############################################################

# Create VTK files for visualization output
vtkfile_1 = XDMFFile(MPI.comm_world,"reaction_system/Mn.xdmf")
vtkfile_1.parameters["flush_output"] = True
vtkfile_2 = XDMFFile(MPI.comm_world,"reaction_system/M.xdmf")
vtkfile_2.parameters["flush_output"] = True
vtkfile_3 = XDMFFile(MPI.comm_world,"reaction_system/Tn.xdmf")
vtkfile_3.parameters["flush_output"] = True
vtkfile_4 = XDMFFile(MPI.comm_world,"reaction_system/Th.xdmf")
vtkfile_4.parameters["flush_output"] = True
vtkfile_5 = XDMFFile(MPI.comm_world,"reaction_system/Tr.xdmf")
vtkfile_5.parameters["flush_output"] = True
vtkfile_6 = XDMFFile(MPI.comm_world,"reaction_system/Tc.xdmf")
vtkfile_6.parameters["flush_output"] = True
vtkfile_7 = XDMFFile(MPI.comm_world,"reaction_system/Dn.xdmf")
vtkfile_7.parameters["flush_output"] = True
vtkfile_8 = XDMFFile(MPI.comm_world,"reaction_system/D.xdmf")
vtkfile_8.parameters["flush_output"] = True
vtkfile_9 = XDMFFile(MPI.comm_world,"reaction_system/C.xdmf")
vtkfile_9.parameters["flush_output"] = True
vtkfile_10 = XDMFFile(MPI.comm_world,"reaction_system/N.xdmf")
vtkfile_10.parameters["flush_output"] = True
vtkfile_11 = XDMFFile(MPI.comm_world,"reaction_system/Ig.xdmf")
vtkfile_11.parameters["flush_output"] = True
vtkfile_12 = XDMFFile(MPI.comm_world,"reaction_system/mu1.xdmf")
vtkfile_12.parameters["flush_output"] = True
vtkfile_13 = XDMFFile(MPI.comm_world,"reaction_system/mu2.xdmf")
vtkfile_13.parameters["flush_output"] = True
vtkfile_14 = XDMFFile(MPI.comm_world,"reaction_system/H.xdmf")
vtkfile_14.parameters["flush_output"] = True
vtkfile_15 = XDMFFile(MPI.comm_world,"reaction_system/p.xdmf")
vtkfile_15.parameters["flush_output"] = True
vtkfile_16 = XDMFFile(MPI.comm_world,"reaction_system/bfu.xdmf")
vtkfile_16.parameters["flush_output"] = True
#vtkfile_17 = XDMFFile(MPI.comm_world,"reaction_system/ccn.xdmf")
#vtkfile_17.parameters["flush_output"] = True
##############################################################

#VTK file array for saving plots
vtkfile = [vtkfile_1,vtkfile_2,vtkfile_3,vtkfile_4,vtkfile_5,vtkfile_6,vtkfile_7,vtkfile_8,vtkfile_9,\
vtkfile_10,vtkfile_11,vtkfile_12,vtkfile_13,vtkfile_14,vtkfile_15, vtkfile_16]
#,vtkfile_16]
##############################################################

#Mesh and remeshing related info and loop parameters
numCells = mesh.num_cells()
mesh.smooth(100)
Counter=0
t = 0.0
j = int(0)
crvt1, NORMAL1 = Curvature(mesh)
#######################################################################

# calculating V and V0
ccn_0 = maxC*C_n + maxD*D_n + maxDn*Dn_n + maxN*N_n
ccn = maxC*C_n + maxD*D_n + maxDn*Dn_n + maxN*N_n
ccn_total = assemble(ccn*dx)
ccn_total0 = ccn_total
print(ccn_total)

# ways of defining sources, not all are used
bfU = Function(W)
fCenterNeg = Expression('4*exp(-2e4*x[0]*x[0] - 2e4*x[1]*x[1])', degree = 2)
fCenterPos = Expression('exp(-5e5*x[0]*x[0] - 5e5*x[1]*x[1])', degree = 2)
ss3 = Expression('sqrt(x[0]*x[0] + x[1]*x[1])', degree=0)
source3 = conditional(gt(ss3,0.0045),1.0,0)
Source3 = project(source3,S1)
i_source3=np.argwhere(Source3.vector().get_local()[:]<=0)  #making negatives zero
Source3.vector()[i_source3[:,0]] = 1.e-16
f1 = Expression("1", degree = 1)
f0 = Expression("0", degree = 1)
funcf = Expression('sqrt(x[0]*x[0] + x[1]*x[1]) < 0.001 + DOLFIN_EPS ? f1 : f0', f1 = f1, f0 = f0, degree = 2)
f = interpolate(funcf, S1)

for n in range(num_steps):

     if j>=1:
     #     #constructing mechanical problem based on updated RHS, crvt and Normals
         mu = 1
         RHS = RHS_sum(U_n,par)
         RHS_MECH_ = project(RHS,S1)
         bfu, l, p = TrialFunctions(W)
         bfv, w, q = TestFunctions(W)

         FM = (K0 + G0/3)*inner(nabla_div(bfu), nabla_div(bfv))*dx + G0*inner(nabla_grad(bfu), nabla_grad(bfv))*dx - alpha0*inner(p, div(bfv))*dx - (K0*cellVolume*dot(ccn, div(bfv))*dx - K0*cellVolume*dot(ccn_0, div(bfv))*dx)\
         + c0*dot(p, q)*dx - c0*dot(p_n, q)*dx + alpha0*inner(div(bfu), q)*dx - alpha0*inner(div(bfu_n), q)*dx + dt*kappa0*inner(grad(p), grad(q))*dx\
         - sum(l[i]*inner(bfv, Z[i])*dx for i in range(len(Z))) - sum(w[i]*inner(bfu, Z[i])*dx for i in range(len(Z)))

         aM = lhs(FM)
         LM = rhs(FM)
         solve(aM == LM, bfU, bcp)
         bfu, l, p = bfU.split()
         diff1 = ccn - ccn_1
         ##############################################################

         #Loop info update and printing
         print(t,flush=True)
         t+=dt
         ##############################################################
         bfU_n.assign(bfU)
         bfu_n, l_n, p_n = bfU_n.split()

         #############################################################

         crvt1, NORMAL1 = Curvature(mesh)
         #mmu_n.assign(project(dis,VV1))

         bfu.rename('bfu', 'bfu')
         vtkfile[15].write(bfu,t)
         p.rename('p', 'p')
         vtkfile[14].write(p,t)
         ##############################################################

         ccn_1 = ccn

         # #Update biology PDE and solve
         F1 = ((Mn - Mn_n)/k)*v1*dx - (par[45] - par[24]*Mn - par[49]*Mn*(par[0]*Ig + par[1]*mu1))*v1*dx\
         + ((M - M_n)/k)*v2*dx + D_cell*dot(grad(M), grad(v2))*dx + coeff*M*DarcyConst*div(grad(p_n))*v2*dx - ((-par[25])*M + Mn*(par[0]*Ig + par[1]*mu1))*v2*dx\
         + ((Th - Th_n)/k)*v4*dx + D_cell*dot(grad(Th), grad(v4))*dx + coeff*Th*DarcyConst*div(grad(p_n))*v4*dx - (Tn*(par[2]*M + par[3]*D) - Th*(par[29] + par[27]*Tr + par[28]*mu1))*v4*dx\
         + ((Tr - Tr_n)/k)*v5*dx + D_cell*dot(grad(Tr), grad(v5))*dx + coeff*Tr*DarcyConst*div(grad(p_n))*v5*dx - ((-par[30])*Tr + par[4]*Tn*mu1)*v5*dx\
         + ((Tc - Tc_n)/k)*v6*dx + D_cell*dot(grad(Tc), grad(v6))*dx + coeff*Tc*DarcyConst*div(grad(p_n))*v6*dx - (Tn*(par[6]*M + par[5]*Th + par[7]*D) - Tc*(par[33] + par[31]*Tr + par[32]*mu1))*v6*dx - 10.0*fCenterPos*v6*dx\
         + ((Tn - Tn_n)/k)*v3*dx - (par[46] - par[26]*Tn - par[50]*Tn*(par[2]*M + par[3]*D) - par[52]*Tn*(par[6]*M + par[5]*Th + par[7]*D) - par[4]*par[51]*Tn*mu1)*v3*dx\
         + ((Dn - Dn_n)/k)*v7*dx - (par[47] - par[34]*Dn - par[53]*Dn*(par[8]*C + par[9]*H))*v7*dx\
         + ((D - D_n)/k)*v8*dx + D_cell*dot(grad(D), grad(v8))*dx + coeff*D*DarcyConst*div(grad(p_n))*v8*dx - ((-D)*(par[36] + par[35]*C) + Dn*(par[8]*C + par[9]*H))*v8*dx\
         + ((C - C_n)/k)*v9*dx + D_cell*dot(grad(C), grad(v9))*dx + coeff*C*DarcyConst*div(grad(p_n))*v9*dx - ((-C)*(par[39] + par[37]*Tc + par[38]*Ig) + C*(1 - C/par[48])*(2*par[10] + par[11]*mu1 + par[12]*mu2))*v9*dx\
         + ((N - N_n)/k)*v10*dx - ((-par[40])*N + par[54]*C*(par[39] + par[37]*Tc + par[38]*Ig))*v10*dx\
         + ((Ig - Ig_n)/k)*v11*dx + D_cyto*dot(grad(Ig), grad(v11))*dx + coeff*Ig*DarcyConst*div(grad(p_n))*v11*dx - (par[13]*Th + par[14]*Tc - par[41]*Ig)*v11*dx\
         + ((mu1 - mu1_n)/k)*v12*dx + D_cyto*dot(grad(mu1), grad(v12))*dx + coeff*mu1*DarcyConst*div(grad(p_n))*v12*dx - (par[16]*M + par[15]*Th + par[17]*C - par[42]*mu1)*v12*dx\
         + ((mu2 - mu2_n)/k)*v13*dx + D_cyto*dot(grad(mu2), grad(v13))*dx + coeff*mu2*DarcyConst*div(grad(p_n))*v13*dx - (par[19]*M + par[18]*Th + par[20]*C - par[43]*mu2)*v13*dx\
         + ((H - H_n)/k)*v14*dx + D_H*dot(grad(H), grad(v14))*dx + coeff*H*DarcyConst*div(grad(p_n))*v14*dx - (par[21]*M + par[22]*D + par[23]*N - par[44]*H)*v14*dx

         # Case of Robin Boundary Condition
         # F1 = ((Mn - Mn_n)/k)*v1*dx - (par[45] - par[24]*Mn - par[49]*Mn*(par[0]*Ig + par[1]*mu1))*v1*dx\
         # + ((M - M_n)/k)*v2*dx + D_cell*dot(grad(M), grad(v2))*dx + coeff*M*DarcyConst*div(grad(p_n))*v2*dx + 100*D_cell*dot(M, v2)*ds - D_cell*dot(1.0, v2)*ds - ((-par[25])*M + Mn*(par[0]*Ig + par[1]*mu1))*v2*dx\
         # + ((Tn - Tn_n)/k)*v3*dx - (par[46] - par[26]*Tn - par[50]*Tn*(par[2]*M + par[3]*D) - par[52]*Tn*(par[6]*M + par[5]*Th + par[7]*D) - par[4]*par[51]*Tn*mu1)*v3*dx\
         # + ((Th - Th_n)/k)*v4*dx + D_cell*dot(grad(Th), grad(v4))*dx + coeff*Th*DarcyConst*div(grad(p_n))*v4*dx + 100*D_cell*dot(Th, v4)*ds - D_cell*dot(1.0, v4)*ds- (Tn*(par[2]*M + par[3]*D) - Th*(par[29] + par[27]*Tr + par[28]*mu1))*v4*dx\
         # + ((Tr - Tr_n)/k)*v5*dx + D_cell*dot(grad(Tr), grad(v5))*dx + coeff*Tr*DarcyConst*div(grad(p_n))*v5*dx + 100*D_cell*dot(Tr, v5)*ds - D_cell*dot(1.0, v5)*ds - ((-par[30])*Tr + par[4]*Tn*mu1)*v5*dx\
         # + ((Tc - Tc_n)/k)*v6*dx + D_cell*dot(grad(Tc), grad(v6))*dx + coeff*Tc*DarcyConst*div(grad(p_n))*v6*dx + 100*D_cell*dot(Tc, v6)*ds - D_cell*dot(1.0, v6)*ds - (Tn*(par[6]*M + par[5]*Th + par[7]*D) - Tc*(par[33] + par[31]*Tr + par[32]*mu1))*v6*dx\
         # + ((Dn - Dn_n)/k)*v7*dx + D_cell*dot(grad(Dn), grad(v7))*dx + coeff*Dn*DarcyConst*div(grad(p_n))*v7*dx + 100*D_cell*dot(Dn, v7)*ds - D_cell*dot(1.0, v7)*ds - (par[47] - par[34]*Dn - par[53]*Dn*(par[8]*C + par[9]*H))*v7*dx\
         # + ((D - D_n)/k)*v8*dx + D_cell*dot(grad(D), grad(v8))*dx + coeff*D*DarcyConst*div(grad(p_n))*v8*dx - ((-D)*(par[36] + par[35]*C) + Dn*(par[8]*C + par[9]*H))*v8*dx\
         # + ((C - C_n)/k)*v9*dx + D_cell*dot(grad(C), grad(v9))*dx + coeff*C*DarcyConst*div(grad(p_n))*v9*dx - ((-C)*(par[39] + par[37]*Tc + par[38]*Ig) + C*(1 - C/par[48])*(2*par[10] + par[11]*mu1 + par[12]*mu2))*v9*dx\
         # + ((N - N_n)/k)*v10*dx - ((-par[40])*N + par[54]*C*(par[39] + par[37]*Tc + par[38]*Ig))*v10*dx\
         # + ((Ig - Ig_n)/k)*v11*dx + D_cyto*dot(grad(Ig), grad(v11))*dx + coeff*Ig*DarcyConst*div(grad(p_n))*v11*dx - (par[13]*Th + par[14]*Tc - par[41]*Ig)*v11*dx\
         # + ((mu1 - mu1_n)/k)*v12*dx + D_cyto*dot(grad(mu1), grad(v12))*dx + coeff*mu1*DarcyConst*div(grad(p_n))*v12*dx - (par[16]*M + par[15]*Th + par[17]*C - par[42]*mu1)*v12*dx\
         # + ((mu2 - mu2_n)/k)*v13*dx + D_cyto*dot(grad(mu2), grad(v13))*dx + coeff*mu2*DarcyConst*div(grad(p_n))*v13*dx - (par[19]*M + par[18]*Th + par[20]*C - par[43]*mu2)*v13*dx\
         # + ((H - H_n)/k)*v14*dx + D_H*dot(grad(H), grad(v14))*dx + coeff*H*DarcyConst*div(grad(p_n))*v14*dx - (par[21]*M + par[22]*D + par[23]*N - par[44]*H)*v14*dx

         bcB = []
         # a = lhs(F1)
         # L = rhs(F1)
         # solve(a==L, U, bcB)
         solve(F1==0, U, bcB)
         ##############################################################

         ##############################################################
         Mn_,M_,Tn_,Th_,Tc_,Tr_,Dn_,D_,C_,N_,Ig_,mu1_,mu2_,H_= U.split()
         ##############################################################

         #Saving info of the previous time step
         U_n.assign(U)
         Mn_n,M_n,Tn_n,Th_n,Tr_n,Tc_n,Dn_n,D_n,C_n,N_n,Ig_n,mu1_n,mu2_n,H_n= U_n.split()
         #######################################################################
         ##############################################################
         ccn = maxC*C_n + maxD*D_n + maxDn*Dn_n + maxN*N_n
         ccn_total = assemble(ccn*dx)
         ratio = ccn_total/assemble(ccn_0*dx)
         print('ccntotal = ', ccn_total, 'ratio = ', ratio)#, 'density = ', ccn_total/area)
     if j%10==0:
#         mesh.smooth(100)
         Mn_n.rename('Mn_n','Mn_n')
         M_n.rename('M_n','M_n')
         Tn_n.rename('Tn_n','Tn_n')
         Th_n.rename('Th_n','Th_n')
         Tr_n.rename('Tr_n','Tr_n')
         Tc_n.rename('Tc_n','Tc_n')
         Dn_n.rename('Dn_n','Dn_n')
         D_n.rename('D_n','D_n')
         C_n.rename('C_n','C_n')
         N_n.rename('N_n','N_n')
         Ig_n.rename('Ig_n','Ig_n')
         mu1_n.rename('mu1_n','mu1_n')
         mu2_n.rename('mu2_n','mu2_n')
         H_n.rename('H_n','H_n')
         vtkfile[0].write(Mn_n,t)
         vtkfile[1].write(M_n,t)
         vtkfile[2].write(Tn_n,t)
         vtkfile[3].write(Th_n,t)
         vtkfile[4].write(Tr_n,t)
         vtkfile[5].write(Tc_n,t)
         vtkfile[6].write(Dn_n,t)
         vtkfile[7].write(D_n,t)
         vtkfile[8].write(C_n,t)
         vtkfile[9].write(N_n,t)
         vtkfile[10].write(Ig_n,t)
         vtkfile[11].write(mu1_n,t)
         vtkfile[12].write(mu2_n,t)
         vtkfile[13].write(H_n,t)

         CRVT = project(crvt1,S1)
         NORM = project(NORMAL1,VV1)
#         CRVT.rename('CRVT','CRVT')
#         NORM.rename('NORM','NORM')
         #           vtkfile_34.write(CRVT,t)
         #          vtkfile_35.write(NORM,t)
         ##############################################################

         #Plotting the integrals every 100 steps so that it doesn't take much time
         ########################################s######################
         # if j%100==0:
     # if j == num_steps - 2:
	 #     displ = project(bfu,VV1)
	 #     ALE.move(mesh,displ)
     j+=1
     #Remeshing
     if j%remesh_step==0:
         a_1 = project(Constant(1),R)
         Domain_vol = assemble(a_1*dx)
         CellArea = Domain_vol/(numCells)  #add it to (j/10) is to make the mesh one cell finer each time
         MeshSize = sqrt(4*CellArea/sqrt(3))
         Counter+=1
         GeoMaker(MeshSize,mesh,'Mesh1',Refine,Counter)
         mesh = Mesh("Mesh1.xml")
         numCells = mesh.num_cells()
         print(numCells)
         while numCells > int(Max_cellnum):
             MeshSize+= MeshSize/100
             GeoMaker(MeshSize,mesh,'Mesh1',Refine,Counter)
             mesh = Mesh("Mesh1.xml")
             numCells = mesh.num_cells()
             print(numCells)
         while (numCells < int(Min_cellnum)) and (numCells < int(Max_cellnum)):
             MeshSize-= MeshSize/200
             GeoMaker(MeshSize,mesh,'Mesh1',Refine,Counter)
             mesh = Mesh("Mesh1.xml")
             numCells = mesh.num_cells()
             print(numCells)
         Volume = MeshFunction("size_t", mesh, "Mesh1_physical_region.xml")
         bnd_mesh = MeshFunction("size_t", mesh, "Mesh1_facet_region.xml")
         xdmf = XDMFFile(mesh.mpi_comm(),"Mesh1.xdmf")
         xdmf.write(mesh)
         xdmf.write(Volume)
         xdmf = XDMFFile(mesh.mpi_comm(),"boundaries1.xdmf")
         xdmf.write(bnd_mesh)
         xdmf.close()

         mesh = Mesh()
         xdmf = XDMFFile(mesh.mpi_comm(), "Mesh1.xdmf")
         xdmf.read(mesh)

         mvc = MeshValueCollection("size_t", mesh, 2)
         with XDMFFile("Mesh1.xdmf") as infile:
             infile.read(mvc, "f")
         Volume = cpp.mesh.MeshFunctionSizet(mesh, mvc)
         xdmf.close()

         mvc2 = MeshValueCollection("size_t", mesh, 1)
         with XDMFFile("boundaries1.xdmf") as infile:
             infile.read(mvc2, "f")
         bnd_mesh = cpp.mesh.MeshFunctionSizet(mesh, mvc2)
         ###############################################################
         # Build function space
         P22 = VectorElement("P", mesh.ufl_cell(), 4)
         P00 = VectorElement("R", mesh.ufl_cell(), 0,dim=3)
         Q = FiniteElement("P", mesh.ufl_cell(), 1)
         element = MixedElement([P22, P00, Q])
         W = FunctionSpace(mesh, element)
         S1 = FunctionSpace(mesh,'P',1)
         S2 = FunctionSpace(mesh,'P',2)
         VV = VectorFunctionSpace(mesh,'Lagrange',4)
         VV1 = VectorFunctionSpace(mesh,'Lagrange',1)
         R = FunctionSpace(mesh,'R',0)
         P1 = FiniteElement('P', triangle,1)
         P3 = FiniteElement('P', triangle,1)
         PB = FiniteElement('B', triangle,3)
         NEE = NodalEnrichedElement(P1, PB)
         element1 = MixedElement([NEE,NEE,NEE,NEE,NEE,NEE,NEE,NEE,NEE,NEE,NEE,NEE,NEE,NEE])
         #element1 = MixedElement([P1,P3,P3,P3,P3,P3,P1,P3,P3,P3,P3,P1,P1,P1,P1])
         Mixed_Space1 = FunctionSpace(mesh, element1)
         ###############################################################


         #Defining functions and test functions
         U = Function(Mixed_Space1)
         U_n.set_allow_extrapolation(True)
         U_n = interpolate(U_n,Mixed_Space1)
         Mn, M, Tn, Th, Tr, Tc, Dn, D, C, N, Ig, mu1, mu2, H = split(U)
         Mn_n, M_n, Tn_n, Th_n, Tr_n, Tc_n, Dn_n, D_n, C_n, N_n, Ig_n, mu1_n, mu2_n, H_n = U_n.split()
         v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14= TestFunctions(Mixed_Space1)
         # bfU.set_allow_extrapolation(True)
         # bfU = interpolate(bfU, W)
         bfU = Function(W)
         bfU_n.set_allow_extrapolation(True)
         bfU_n = interpolate(bfU_n, W)
         # bfu, l, p = split(bfU)
         bfu_n, l_n, p_n = bfU_n.split()
         vel = Function(VV)
         crvt1, NORMAL1 = Curvature(mesh)
         #######################################################################

         # Construct integration measure using these markers
         ds = Measure('ds', subdomain_data=bnd_mesh)
         dx = Measure('dx', subdomain_data=Volume)
         ###############################################################
         # bcu = DirichletBC(W.sub(0), Constant((0.0, 0.0)), DomainBoundary())
         # #p_bc = Expression('x[0]*x[1]', degree = 1)
         bcp = DirichletBC(W.sub(2), Constant(0.0), DomainBoundary())
         # bc = [bcu, bcp]
         ccn = maxC*C_n + maxD*D_n + maxDn*Dn_n + maxN*N_n
         Ctl = 1
         print('Remeshing done!')

     #######################################################################

list_timings(TimingClear.clear, [TimingType.wall])
