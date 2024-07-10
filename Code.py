import numpy as np
import math


# STEP 1: Reading Nuclear Repulsion Energy
fenuc = open("enuc.dat","r")

#STEP 2: Reading Overlap Matrix,Kinetic Energy and Nuclear attraction integrals; forming the core Hamiltonian

fs = open("s.dat","r")
fke = open("ke.dat","r")
fv = open("pe_v.dat","r")

hc = np.zeros((7,7))

for k in fke:
  kl=k.split()
  hc[int(kl[0])-1][int(kl[1])-1] = float(kl[2])
  if int(kl[0])!=int(kl[1]):
    hc[int(kl[1])-1][int(kl[0])-1] = float(kl[2])

for p in fv:
  pl=p.split()
  hc[int(pl[0])-1][int(pl[1])-1] += float(pl[2])
  if int(pl[0])!=int(pl[1]):
    hc[int(pl[1])-1][int(pl[0])-1] += float(pl[2])

#print(hc)

# STEP 3: Reading the two electron integral and storing them in a 1D array 

ftei = open("two_integ.dat","r")
tei = np.zeros((406))

for t in ftei:
  tl=t.split()
  for i in range(4):
    tl[i] = int(tl[i])
  tl[4] = float(tl[4]) 
  b = int(tl[0]-1)*(int(tl[0]-1) + 1)/2 + int(tl[1]-1)
  c = int(tl[2]-1)*(int(tl[2]-1) + 1)/2 + int(tl[3]-1)
  if b >= c:
    a = int( b*(b+1)/2 + c )
  else :
    a = int( c*(c+1)/2 + b)
  #print (a)  
  tei[a] = tl[4]

#print(tei)


# STEP 4: Building the Orthogonalization Matrix

# Building Overlap Matrix S by reading the file
s = np.zeros((7,7))
for a in fs:
  sl=a.split()
  s[int(sl[0])-1][int(sl[1])-1] = float(sl[2])
  if int(sl[0])!=int(sl[1]):
    s[int(sl[1])-1][int(sl[0])-1] = float(sl[2])
  #print(float(sl[2]))
#print(s)

seval,seigv = np.linalg.eigh(s)   # Finding the eigenvalues and eigenvectors of Overlap Matrix S

#print(seval)
#print(seigv)

s_diag_irt = np.zeros((7,7))
for i in range(7):
  s_diag_irt[i][i] = 1/(math.sqrt(seval[i]))  # Building s^(-1/2); s being the diagonalized matrix of S 

s_inv_half = np.zeros((7,7))
a = np.matmul(s_diag_irt,seigv.transpose())
s_inv_half = np.matmul(seigv,a)               # Building the Orthogonalization Matrix S^(-1/2) 

#print(s_inv_half)  


# STEP 5: Building the initial guess density matrix

fprime = np.zeros((7,7))
a = np.matmul(hc,s_inv_half)
fprime = np.matmul(s_inv_half.transpose(),a)   # Building the initial (guess) fock matrix
print('Fock Matrix:')
print(fprime)

orb_eng,cprime= np.linalg.eigh(fprime)   # cprime (the coefficients) are the coefficients of the eigenvectors (C0 prime).
c0 = np.zeros((7,7))
c0 = np.matmul(s_inv_half,cprime)        # Transforming the eigenvectors into the original (non-orthogonal) AO basis

#print('cprime:')
#print(cprime)
#print('c0:')
#print(c0)

d = np.zeros((7,7))
for i in range(7):
  for j in range(7):
    for k in range(5):             # k loops till range(5),because we're summing only over the occupied MOs.
      d[i][j]+=c0[i][k]*c0[j][k]   # Building the density matrix using the occupied MOs

#print('Density Matrix:')
#print(d)

# STEP 6: Computing the initial SCF Energy

e_elec0 = 0.0

for i in range(7):
  for j in range(7):
    e_elec0 += d[i][j]*(hc[i][j]) + d[i][j]*(hc[i][j])  # SCF electronic energy computed using the density matrix 
#print('e_elec0:',e_elec0)

enuc = float(fenuc.read())
#print('enuc:',enuc)
e_tot0 = e_elec0 + enuc       # Total energy = Sum of the electronic energy and the nuclear repulsion energy
#print(e_tot0)


# STEP 7: Computing the New Fock Matrix

def index(a,b):           # Function to compute the compound indices
  if a >= b:
    return a*(a+1)/2 + b
  else :
    return b*(b+1)/2 + a

def new_f(d,tei):
    fnew = np.zeros((7,7))
    for i in range(7):
      for j in range(7):
        fnew[i][j] = hc[i][j]
        for k in range(7):
          for l in range(7):
            ij = index(i,j)
            ik = index(i,k)
            kl = index(k,l) 
            jl = index(j,l)
            ijkl = int(index(ij,kl))
            ikjl = int(index(ik,jl))
            fnew[i][j] += d[k][l]*(2.0*tei[ijkl] - tei[ikjl])  # Building the new Fock matrix using previous iteration density

    #print('New Fock Matrix:')
    #print(fnew)
    return fnew
    #new_d(fnew)



# STEP 8: Building the new density matrix 
def new_d(fnew):
    a = np.matmul(fnew,s_inv_half)
    fi = np.matmul(s_inv_half.transpose(),a)
    #print('Fock Matrix (i):')
    #print(fi)

    orb_eng  , cprime= np.linalg.eigh(fi)   # cprime (the coefficients) is the coefficient of the eigenvector (C0 prime).
    c0 = np.zeros((7,7))
    c0 = np.matmul(s_inv_half,cprime)
    """
    print('cprime:')
    print(cprime)
    print('c0:')
    print(c0)
    """
    di = np.zeros((7,7))
    for i in range(7):
      for j in range(7):
        for k in range(5):
          di[i][j]+=c0[i][k]*c0[j][k]     # Building current iteration density matrix

    #print('Density Matrix:')
    #print(di)
    return di



# STEP 9: Computing the new SCF Energy

def new_energy(d,hc,fnew):
    e_eleci = 0

    for i in range(7):
      for j in range(7):
        e_eleci += d[i][j]*(hc[i][j] + fnew[i][j])  # Electronic energy of the ith iteration

    fenuc = open("/content/drive/MyDrive/Hartree Fock/enuc.dat","r")
    enuc = float(fenuc.read())

    e_toti = e_eleci + enuc    # Total energy of the ith iteration 
    return e_toti , e_eleci


# STEP 10: Testing for Convergence

e_toti_prev = e_tot0 

delta1 = 1e-12
delta2 = 1e-11
ctr = 0
print("%s    %s            %s          %s        %s" %('Iter','E(elec)','E(tot)','Delta(E)','RMS(D)'))
print("%2d %15.12f %15.12f" %(ctr,e_elec0,e_tot0))
while True :
  fnew = new_f(d,tei)
  di = new_d(fnew)
  e_toti , e_eleci = new_energy(di,hc,fnew)
 
  delE = e_toti - e_toti_prev
  rmsd = 0
  for i in range(7):
    for j in range(7):
      rmsd += (di[i][j] - d[i][j])*(di[i][j] - d[i][j]) # Can also be coded as rmsd+= (di[i][j] - d[i][j])**2
  
  rmsd = math.sqrt(rmsd)
  ctr+=1
  print("%2d %15.12f  %15.12f  %15.12f %13.12f" %(ctr,e_eleci,e_toti,delE,rmsd))
  if abs(delE) < delta1 and rmsd < delta2 :   # Checking the energy difference and root-mean-squared-difference for convergence
      break
  e_toti_prev = e_toti
  d = di
