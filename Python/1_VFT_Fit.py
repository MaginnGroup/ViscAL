# -*- coding: utf-8 -*-
"""
Script to display the viscosity data and the corresponding VFT fits.

Sections:
    . Imports
    . Configuration
    . Auxiliary Functions
        . VFT()
    . Load Data
    . Main Script
    . Plots

Last edit: 2024-05-22
Author: Dinis Abranches
"""

# =============================================================================
# Imports
# =============================================================================

# General
import os

# Specific
import numpy
import pandas
from scipy import optimize
from matplotlib import pyplot as plt

# =============================================================================
# Configuration
# =============================================================================

# Path to database folder
dbPath=r'/path/to/Databases'
# Database code
code='ChCl_EG_H2O' # 'ChCl_EG_H2O','ChCl_EG_ACN','ChCl_EG_DMSO',
                   # 'ChCl_AN1_H2O',ChCl_AN1_ACN,ChCl_AN1_DMSO
                   # 'ChCl_AN2_H2O','ChCl_AN2_ACN','ChCl_AN2_DMSO'
# Turn off axis
axisX=True
axisY=True

# =============================================================================
# Auxiliary Functions
# =============================================================================

def VFT(T,eta_0,B,T_0):
    """
    VFT() computes viscosity at temperature T using the VFT equation.

    Parameters
    ----------
    T : numpy array
        Array of temperatures where viscosities are calculated.
    eta_0 : float
        VFT parameter.
    B : float
        VFT parameter.
    T_0 : float
        VFT parameter.

    Returns
    -------
    eta : numpy array
        Calculated viscosities.

    """
    eta=eta_0*numpy.exp(B/(T-T_0))
    # Output
    return eta

# =============================================================================
# Load Data
# =============================================================================

# Load binary data
# Path to binary CSV file
if 'AN1' in code:
    newCode=code.replace('AN1','AN').split('_')[:2]
elif 'AN2' in code:
    newCode=code.replace('AN2','AN').split('_')[:2]
else:
    newCode=code.split('_')[:2]
newCode='_'.join(newCode)
trainDB_Path=os.path.join(dbPath,newCode+'.csv')
# Load CSV file
trainDB=pandas.read_csv(trainDB_Path)
# Get mole ratio list
tempList=trainDB.iloc[:,:2].values.tolist()
# Get unique mole ratios
ratioList=[]
for entry in tempList:
    if entry not in ratioList: ratioList.append(entry)
# Remove ratios not requested
if 'AN1' in code: del ratioList[-1]
elif 'AN2' in code: del ratioList[0]
# Format as legends
legends_binary=ratioList.copy()
for n in range(len(legends_binary)):
    legends_binary[n]=str(legends_binary[n][0])+':'\
        +str(legends_binary[n][1])+':0'
# Get X and Y tensors
X_binary=numpy.zeros((1,6,1))
Y_binary=numpy.zeros((1,6,1))
mask=trainDB['mol 2']==ratioList[0][1]
X_binary[0]=trainDB[mask].iloc[:,4].to_numpy().reshape(-1,1).copy()
Y_binary[0]=trainDB[mask].iloc[:,5].to_numpy().reshape(-1,1).copy()

# Load ternary data
# Path to ternary CSV file
if 'AN1' in code:
    newCode=code.replace('AN1','AN')
elif 'AN2' in code:
    newCode=code.replace('AN2','AN')
else:
    newCode=code
trainDB_Path=os.path.join(dbPath,newCode+'.csv')
# Load CSV file
trainDB=pandas.read_csv(trainDB_Path)
# Get mole ratio list
tempList=trainDB.iloc[:,:3].values.tolist()
# Get unique mole ratios
ratioList=[]
for entry in tempList:
    if entry not in ratioList: ratioList.append(entry)
# Remove ratios not requested
if 'AN1' in code: del ratioList[-4:]
elif 'AN2' in code: del ratioList[:4]
# Format as legends
legends=ratioList.copy()
for n in range(len(legends)):
    legends[n]=[round(aux) if aux!=0.5 else aux for aux in legends[n]]
    legends[n]=str(legends[n][0])+':'+str(legends[n][1])+':'+str(legends[n][2])
# Get X and Y tensors
X_ternary=numpy.zeros((4,6,1))
Y_ternary=numpy.zeros((4,6,1))
baseMask=trainDB['mol 2']==ratioList[0][1]
for n in range(4):
    mask=numpy.logical_and(baseMask,trainDB['mol 3']==ratioList[n][2])
    X_ternary[n]=trainDB[mask].iloc[:,6].to_numpy().reshape(-1,1)
    Y_ternary[n]=trainDB[mask].iloc[:,7].to_numpy().reshape(-1,1)

# =============================================================================
# Main Script
# =============================================================================

# Concatenate binary and ternary data
legends=legends_binary+legends
X=numpy.concatenate((X_binary,X_ternary),axis=0)
Y=numpy.concatenate((Y_binary,Y_ternary),axis=0)

# Initialize VFT parameters array
VFTpar=numpy.zeros((X.shape[0],3))
# Perform fitting
for n in range(X.shape[0]):
    VFTpar[n,:]=optimize.curve_fit(VFT,X[n].reshape(-1,),Y[n].reshape(-1,),
                                   p0=[0.01,1000,150],
                                   bounds=([0.01,200,100],[0.2,2000,200]),
                                   sigma=0.01*numpy.ones((Y[n].shape[0],)),
                                   maxfev=10000)[0]

# =============================================================================
# Plots
# =============================================================================

# Pyplot Configuration
plt.rcParams['figure.dpi']=600
plt.rcParams['savefig.dpi']=600
plt.rcParams['text.usetex']=False
plt.rcParams['font.family']='serif'
plt.rcParams['font.serif']='Times New Roman'
plt.rcParams['font.weight']='bold'
plt.rcParams['mathtext.rm']='serif'
plt.rcParams['mathtext.it']='serif:italic'
plt.rcParams['mathtext.bf']='serif:bold'
plt.rcParams['mathtext.fontset']='custom'
plt.rcParams['axes.titlesize']=8
plt.rcParams['axes.labelsize']=8
plt.rcParams['xtick.labelsize']=7
plt.rcParams['ytick.labelsize']=7
plt.rcParams['font.size']=8
plt.rcParams["savefig.pad_inches"]=0.02

# Composition slices
colors=['k','b','r','g','y']
plt.figure(figsize=(2.3,2))
MREList=[]
for n in range(X.shape[0]):
    plt.plot(X[n].reshape(-1,),Y[n].reshape(-1,),'o'+colors[n],
             label=legends[n],markersize=3)
    plt.plot(X[n].reshape(-1,),VFT(X[n].reshape(-1,),*VFTpar[n,:]),
             '--'+colors[n],linewidth=1)
    MRE=100*numpy.abs(VFT(X[n].reshape(-1,),*VFTpar[n,:])-Y[n].reshape(-1,))
    MRE=MRE/Y[n].reshape(-1,)
    MREList.append(MRE.mean())
plt.text(0.01,0.92,'MRE='+'{:.1f}'.format(numpy.mean(MREList))+'%',
         horizontalalignment='left',transform=plt.gca().transAxes,c='r')
plt.ylim([0,40])
plt.legend(fontsize=6)
plt.title(':'.join(newCode.split('_')),weight='bold')
if not axisX:
    plt.gca().get_xaxis().set_ticklabels([])
else:
    plt.xlabel('Temperature /K',weight='bold')
if not axisY:
    plt.gca().get_yaxis().set_ticklabels([])
else:
    plt.ylabel('Viscosity /cP',weight='bold')
plt.show()