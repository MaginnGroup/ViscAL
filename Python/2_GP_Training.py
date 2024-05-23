# -*- coding: utf-8 -*-
"""
Script to train a GP on viscosity data for ternary DESs.

Sections:
    . Imports
    . Configuration
    . Auxiliary Functions
        . normalize()
        . buildGP()
        . gpPredict()
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
import warnings

# Specific
import numpy
import pandas
from sklearn import preprocessing
import gpflow
from matplotlib import pyplot as plt
from matplotlib import cm

# =============================================================================
# Configuration
# =============================================================================

# Path to database folder
dbPath=r'/path/to/Databases'
# Database code
code='ChCl_AN2_DMSO' # 'ChCl_EG_H2O','ChCl_EG_ACN','ChCl_EG_DMSO',
                   # 'ChCl_AN1_H2O',ChCl_AN1_ACN,ChCl_AN1_DMSO
                   # 'ChCl_AN2_H2O','ChCl_AN2_ACN','ChCl_AN2_DMSO'
# Turn off y axis
axisY=True
# Define normalization methods
featureNorm='MinMax' # None,Standardization,MinMax
labelNorm='LogStand' # None,Standardization,LogStand
# GP Configuration
gpConfig={'kernel':'RBF',
          'useWhiteKernel':True,
          'trainLikelihood':True}

# =============================================================================
# Auxiliary Functions
# =============================================================================

def normalize(inputArray,skScaler=None,method='Standardization',reverse=False):
    """
    normalize() normalizes (or unnormalizes) inputArray using the method
    specified and the skScaler provided.

    Parameters
    ----------
    inputArray : numpy array
        Array to be normalized. If dim>1, array is normalized column-wise.
    skScaler : scikit-learn preprocessing object or None
        Scikit-learn preprocessing object previosly fitted to data. If None,
        the object is fitted to inputArray.
        Default: None
    method : string, optional
        Normalization method to be used.
        Methods available:
            . Standardization - classic standardization, (x-mean(x))/std(x)
            . MinMax - scale to range (0,1)
            . LogStand - standardization on the log of the variable,
                         (log(x)-mean(log(x)))/std(log(x))
            . Log - simply convert x to log(x)
        Defalt: 'Standardization'
    reverse : bool
        Whether  to normalize (False) or unnormalize (True) inputArray.
        Defalt: False

    Returns
    -------
    inputArray : numpy array
        Normalized (or unnormalized) version of inputArray.
    skScaler : scikit-learn preprocessing object
        Scikit-learn preprocessing object fitted to inputArray. It is the same
        as the inputted skScaler, if it was provided.

    """
    # If inputArray is a labels vector of size (N,), reshape to (N,1)
    if inputArray.ndim==1:
        inputArray=inputArray.reshape((-1,1))
        warnings.warn('Input to normalize() was of shape (N,). It was assumed'\
                      +' to be a column array and converted to a (N,1) shape.')
    # If skScaler is None, train for the first time
    if skScaler is None:
        # Check method
        if method=='Standardization' or method=='MinMax': aux=inputArray
        elif method=='LogStand' or method=='Log': aux=numpy.log(inputArray)
        else: raise ValueError('Could not recognize method in normalize().')
        if method=='MinMax':
            skScaler=preprocessing.MinMaxScaler().fit(aux)
        elif method=='Log':
            skScaler='NA'
        else:
            skScaler=preprocessing.StandardScaler().fit(aux) 
    # Do main operation (normalize or unnormalize)
    if reverse:
        # Rescale the data back to its original distribution
        if method!='Log':
            inputArray=skScaler.inverse_transform(inputArray)
        # Check method
        if method=='LogStand' or method=='Log':
            inputArray=numpy.exp(inputArray)
    elif not reverse:
        # Check method
        if method=='Standardization' or method=='MinMax': aux=inputArray
        elif method=='LogStand' or method=='Log': aux=numpy.log(inputArray)
        else: raise ValueError('Could not recognize method in normalize().')
        if method!='Log':
            inputArray=skScaler.transform(aux)
        else:
            inputArray=aux
    # Return
    return inputArray,skScaler

def buildGP(X_Train,Y_Train,gpConfig={}):
    """
    buildGP() builds and fits a GP model using the training data provided.

    Parameters
    ----------
    X_Train : numpy array (N,K)
        Training features, where N is the number of data points and K is the
        number of independent features (e.g., sigma profile bins).
    Y_Train : numpy array (N,1)
        Training labels (e.g., property of a given molecule).
    gpConfig : dictionary, optional
        Dictionary containing the configuration of the GP. If a key is not
        present in the dictionary, its default value is used.
        Keys:
            . kernel : string
                Kernel to be used. One of:
                    . 'RBF' - gpflow.kernels.RBF()
                    . 'RQ' - gpflow.kernels.RationalQuadratic()
                    . 'Matern32' - gpflow.kernels.Matern32()
                    . 'Matern52' - gpflow.kernels.Matern52()
                The default is 'RQ'.
            . useWhiteKernel : boolean
                Whether to use a White kernel (gpflow.kernels.White).
                The default is True.
            . trainLikelihood : boolean
                Whether to treat the variance of the likelihood of the modeal
                as a trainable (or fitting) parameter. If False, this value is
                fixed at 10^-5.
                The default is True.
        The default is {}.
    Raises
    ------
    UserWarning
        Warning raised if the optimization (fitting) fails to converge.

    Returns
    -------
    model : gpflow.models.gpr.GPR object
        GP model.

    """
    # Unpack gpConfig
    kernel=gpConfig.get('kernel','RQ')
    useWhiteKernel=gpConfig.get('useWhiteKernel','True')
    trainLikelihood=gpConfig.get('trainLikelihood','True')
    # Select and initialize kernel
    if kernel=='RBF':
        gpKernel=gpflow.kernels.SquaredExponential()
    if kernel=='RQ':
        gpKernel=gpflow.kernels.RationalQuadratic()
    if kernel=='Matern32':
        gpKernel=gpflow.kernels.Matern32()
    if kernel=='Matern52':
        gpKernel=gpflow.kernels.Matern52()
    # Add White kernel
    if useWhiteKernel: gpKernel=gpKernel+gpflow.kernels.White()
    # Build GP model    
    model=gpflow.models.GPR((X_Train,Y_Train),gpKernel,noise_variance=10**-2)
    # Select whether the likelihood variance is trained
    gpflow.utilities.set_trainable(model.likelihood.variance,trainLikelihood)
    # Build optimizer
    optimizer=gpflow.optimizers.Scipy()
    # Fit GP to training data
    aux=optimizer.minimize(model.training_loss,
                           model.trainable_variables,
                           method='L-BFGS-B')
    # Check convergence
    if aux.success==False:
        warnings.warn('GP optimizer failed to converge.')
    # Output
    return model

def gpPredict(model,X):
    """
    gpPredict() returns the prediction and standard deviation of the GP model
    on the X data provided.

    Parameters
    ----------
    model : gpflow.models.gpr.GPR object
        GP model.
    X : numpy array (N,K)
        Training features, where N is the number of data points and K is the
        number of independent features (e.g., sigma profile bins).

    Returns
    -------
    Y : numpy array (N,1)
        GP predictions.
    STD : numpy array (N,1)
        GP standard deviations.

    """
    # Do GP prediction, obtaining mean and variance
    GP_Mean,GP_Var=model.predict_f(X)
    # Convert to numpy
    GP_Mean=GP_Mean.numpy()
    GP_Var=GP_Var.numpy()
    # Prepare outputs
    Y=GP_Mean
    STD=numpy.sqrt(GP_Var)
    # Output
    return Y,STD

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

# Concatenate binary and ternary data
legends=legends_binary+legends
X=numpy.concatenate((X_binary,X_ternary),axis=0)
Y=numpy.concatenate((Y_binary,Y_ternary),axis=0)

# =============================================================================
# Main Script
# =============================================================================

# Compute composition matrix
comps=numpy.zeros((len(legends),3))
for n in range(len(legends)):
    mol1,mol2,mol3=[float(aux) for aux in legends[n].split(':')]
    comps[n,0]=mol1/(mol1+mol2+mol3)
    comps[n,1]=mol2/(mol1+mol2+mol3)
    comps[n,2]=mol3/(mol1+mol2+mol3)
# Build X_Train
X_Train=numpy.repeat(comps,X.shape[1],axis=0)
X_Train=numpy.concatenate((X_Train,X.reshape(-1,1)),axis=1)
# Build Y_Train
Y_Train=Y.reshape(-1,1)

# Normalize
X_Train_N=X_Train.copy()
if featureNorm is not None:
    X_Train_N,skScaler_X=normalize(X_Train,method=featureNorm)
if labelNorm is not None:
    Y_Train_N,skScaler_Y=normalize(Y_Train,method=labelNorm)
else:
    Y_Train_N=Y_Train
# Train GP
model=buildGP(X_Train_N,Y_Train_N,gpConfig=gpConfig)
# Get GP predictions
Y_Train_Pred_N,__=gpPredict(model,X_Train_N)
# Unnormalize
if labelNorm is not None:
    Y_Train_Pred,__=normalize(Y_Train_Pred_N,skScaler=skScaler_Y,
                              method=labelNorm,reverse=True)
else:
    Y_Train_Pred=Y_Train_Pred_N
# Compute MRE
MRE=100*numpy.abs(Y_Train_Pred-Y_Train)/Y_Train
MRE=MRE.mean()

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
plt.rcParams['axes.titlesize']=9
plt.rcParams['axes.labelsize']=9
plt.rcParams['xtick.labelsize']=9
plt.rcParams['ytick.labelsize']=9
plt.rcParams['font.size']=8
plt.rcParams["savefig.pad_inches"]=0.02

# Composition slices
colors=['k','b','r','g','y']
plt.figure(figsize=(2.3,2))
for n in range(len(legends)):
    mol1,mol2,mol3=[float(aux) for aux in legends[n].split(':')]
    x1=mol1/(mol1+mol2+mol3)
    x2=mol2/(mol1+mol2+mol3)
    x3=mol3/(mol1+mol2+mol3)
    X_Test=numpy.repeat(numpy.array([x1,x2,x3]).reshape(-1,3),
                        100,axis=0)
    Temp=numpy.linspace(X.min()*0.99,X.max()*1.01,100).reshape(-1,1)
    X_Test=numpy.concatenate((X_Test,Temp),axis=1)
    # Normalize
    X_Test_N=X_Test.copy()
    if featureNorm is not None:
        X_Test_N,__=normalize(X_Test,method=featureNorm,skScaler=skScaler_X)
    # Predict
    Y_Test_Pred_N,__=gpPredict(model,X_Test_N)
    # Unnormalize
    if labelNorm is not None:
        Y_Test_Pred,__=normalize(Y_Test_Pred_N,skScaler=skScaler_Y,
                                 method=labelNorm,reverse=True)
    else:
        Y_Test_Pred=Y_Test_Pred_N
    # Plots
    plt.plot(X[n].reshape(-1,),Y[n].reshape(-1,),'o'+colors[n],
             label=legends[n],markersize=3)
    plt.plot(X_Test[:,3],Y_Test_Pred,'--'+colors[n],linewidth=1)
plt.xlabel('Temperature /K',weight='bold')
plt.ylim([0,40])
plt.legend(fontsize=6)
plt.title(':'.join(newCode.split('_')),weight='bold')
if not axisY:
    plt.gca().get_yaxis().set_ticklabels([])
else:
    plt.ylabel('Viscosity /cP',weight='bold')
plt.text(0.01,0.92,'MRE='+'{:.1f}'.format(MRE)+'%',
         horizontalalignment='left',transform=plt.gca().transAxes,c='r')
plt.show()

# Surface Plot 
mol1,mol2,maxMol3=[float(aux) for aux in legends[-1].split(':')]
mol3=numpy.linspace(0,maxMol3*1.01,100)
x1=(mol1/(mol1+mol2+mol3)).reshape(-1,1)
x2=(mol2/(mol1+mol2+mol3)).reshape(-1,1)
x3=(mol3/(mol1+mol2+mol3)).reshape(-1,1)
T=numpy.linspace(X.min()*0.99,X.max()*1.01,100).reshape(-1,1)
X_Test=numpy.repeat(numpy.concatenate((x1,x2,x3),axis=1).reshape(-1,3),
                    100,axis=0)
X_Test=numpy.concatenate((X_Test,
                          numpy.tile(T,100).flatten(order='F').reshape(-1,1)),
                         axis=1)
# Normalize
X_Test_N=X_Test.copy()
if featureNorm is not None:
    X_Test_N,__=normalize(X_Test,method=featureNorm,skScaler=skScaler_X)
# Predict
Y_Test_Pred_N,__=gpPredict(model,X_Test_N)
# Unnormalize
if labelNorm is not None:
    Y_Test_Pred,__=normalize(Y_Test_Pred_N,skScaler=skScaler_Y,
                             method=labelNorm,reverse=True)
else:
    Y_Test_Pred=Y_Test_Pred_N
fig=plt.figure(figsize=(10,5))
ax=fig.add_subplot(projection='3d',computed_zorder=False)
ax.set_proj_type('persp',focal_length=0.2)
surf=ax.plot_trisurf(X_Test[:,2],X_Test[:,3],Y_Test_Pred.reshape(-1,),
                     cmap=cm.jet,antialiased=False,
                     zorder=4.4)
ax.plot(X_Train[:,2],X_Train[:,3],
        Y_Train[:,0],'ok',zorder=4.6,markersize=3)
ax.set_xlabel('Co-Solvent Mole Fraction',weight='bold')
ax.set_ylabel('Temperature /K',weight='bold')
ax.set_zlabel('Viscosity /cP',weight='bold')
ax.set_box_aspect(None,zoom=0.8)



ax.contour(X_Test[:,2].reshape(100,100,order='F'),
           X_Test[:,3].reshape(100,100,order='F'),
           Y_Test_Pred.reshape(100,100,order='F'),
           zdir='y',offset=ax.get_ylim()[1], cmap='coolwarm')


plt.show()