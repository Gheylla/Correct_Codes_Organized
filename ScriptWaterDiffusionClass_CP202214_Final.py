#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code originally was "HeatFlow Model"created by @author: theimovaara 
on Sun May  9 13:46:11 2021

Then Modified into Water Flux Model (Richard Equation) by Qroup CP2022 14 
for the completion of Assignment 2 _CIE4365-16
"""

import numpy as np
import pandas as pd
import scipy.integrate as spi
import scipy.sparse as sp
import MyTicToc as mt
import matplotlib.pyplot as plt
import seaborn as sns

# Version of code based on classes
import WaterDiffusionClass_CP202214_Final as WF

"""
Script for running Water Diffusion problem. The code is given in the WaterDiffusionClass.
This script, initializes the problem and then calls the solver in the class and it
generates some plots"""

# In[1]: Initialize
sns.set()
# Domain
#the amount of internodes 
nIN = 101

# soil profile until 15 meters depth internodes 
zIN = np.linspace(-1, 0, num=nIN).reshape(nIN, 1)   


#defining the internodes 
zN = np.zeros(nIN - 1).reshape(nIN - 1, 1)
zN[0, 0] = zIN[0, 0]                                                    #top
zN[1:nIN - 2, 0] = (zIN[1:nIN - 2, 0] + zIN[2:nIN - 1, 0]) / 2          #middle
zN[nIN - 2, 0] = zIN[nIN - 1]                                           #last                    
nN = np.shape(zN)[0]                                                    #number of nodes. 

ii = np.arange(0, nN - 1)
dzN = (zN[ii + 1, 0] - zN[ii, 0]).reshape(nN - 1, 1)                    #height between nodes
dzIN = (zIN[1:, 0] - zIN[0:-1, 0]).reshape(nIN - 1, 1)                  #height between internodes

# collect model dimensions in a namedtuple: modDim
mDim = {'zN' : zN,
        'zIN' : zIN,
        'dzN' : dzN,
        'dzIN' : dzIN,
        'nN' : nN,
        'nIN' : nIN
        }
mDim = pd.Series(mDim)

# collect soil parameters in a pandas Series: sPar
sPar = {'vGA': np.ones(np.shape(zN)) * 1 / 0.5,  # alpha[1/m]
        'vGN': np.ones(np.shape(zN)) * 3.0,  # n[-]
        'vGM': np.ones(np.shape(zN)) * (1 - 1 / 3.0),  # m = 1-1/n[-]
        'thetaSat': np.ones(np.shape(zN)) * 0.4,  # saturated water content
        'thetaRes': np.ones(np.shape(zN)) * 0.03,  # residual water content
        'KSat': np.ones(np.shape(zN)) * 0.05,  # [m/day]
        'vGE': 0.5,  # power factor for Mualem-van Genuchten                      
        'Cv': 1.0e-8,  # compressibility of compact sand [1/Pa]
        }
sPar = pd.Series(sPar)

# In[2] Boundary parameters
# Define top boundary condition function for Water Flux
# =============================================================================
bPar = {'qTop': -0.001,  # top flux
        'WatBotCond': 'robins', # Robin condition or Gravity condition
        'BotRobResTerm': 0.005 * 86400,  # Robin resistance term for bottom was per second but needs to be per day
        'externalrobhead' : -1, # pressure head at lower boundary robins  
        }

bPar = pd.Series(bPar)

# Initial Conditions
zRef = -0.75 # Depth of Water, phreatic water level is 25cm so 25 cm below surface is at -0.25. 
hwIni = zRef - zN
MyWF = WF.WaterDiffusion(sPar, mDim, bPar)

# In[3]: Solve IVP over time range

# Time Discretization
tOut = np.linspace(0,  225, 22500)  # time 225 days in total, 25 zero flux and 200 constant flux
nOut = np.shape(tOut)[0]
print('Solving unsaturated water flow problem')

mt.tic()
result = MyWF.IntegratehwF(tOut, hwIni.squeeze(), nOut)
t = result.t
mt.toc()


# Dirichlet boundary condition: write boundary temperature to output.
if result.success:
    print('Integration has been successful')

qw = MyWF.WaterFlux(tOut, result.y, nOut)
ThetaOut = MyWF.ThetaFun(result.y)

# =============================================================================
# for i in range(nOut):
#     if t[i]<= 25:  # days
#         result.y[0, nIN - 1] = 0
#     else:
#         result.y[0, nIN - 1] = bPar.qTop
# =============================================================================
        


# In[4] Water Flux vs Time Figures: 
plt.close('all')
##Figure 1
fig1, ax1 = plt.subplots(figsize=(7, 4))

for ii in np.arange(0, nN, 10):
    ax1.plot(t, result.y[ii, :], '-')
    
    
ax1.set_title('Water Flux (ODE)')
ax1.set_xlabel('time (days)')
ax1.set_ylabel('Water Flux [m]')

##Figure 2
fig2, (ax21,ax22,ax23) = plt.subplots(3,1,figsize=(10, 7))
ii = 0
for ii in np.arange(0, nN,10):
    ax21.plot(result.t[1:2500], result.y[ii, 1:2500], '-')
    ax22.plot(result.t[2500:3000], result.y[ii, 2500:3000], '-')
    ax23.plot(result.t[3000:22500], result.y[ii, 3000:22500], '-')
ax21.set_title('0-25d')
ax22.set_title('25-30d')
ax23.set_title('30-225d')
fig2.suptitle('Water head (m) vs. Time (d)')
ax21.grid(b=True)
ax22.grid(b=True)
ax23.grid(b=True)

# In[5] Water Flux vs Depth Figures: 
    
fig3, ax3 = plt.subplots(figsize=(4, 7))
for ii in np.arange(0, nOut, 10):
    ax3.plot(result.y[:, ii], zN, '-')

ax3.set_title('Water Flux vs. depth (ODE)')
ax3.set_ylabel('depth [m]')
ax3.set_xlabel('Water Flux [m]')


fig4, ax4 = plt.subplots(figsize=(4, 7))
# plot fluxes after 2nd output time (initial rate is extreme due to initial conditions)
for ii in np.arange(2, nOut, 10):
    ax4.plot(qw[:, ii], zIN, '-')

ax4.set_title('Water Flux vs. depth (ODE)')
ax4.set_ylabel('depth [m]')
ax4.set_xlabel('Water Flux []')

fig5, (ax51,ax52,ax53) = plt.subplots(1,3,figsize=(8, 6))
ii = 0
for ii,t in enumerate((result.t[0:len(result.t):10]).tolist()):
    if t<25:
        ax51.plot(result.y[:, ii*10], zN, '-')
    if 25<=t<30:
        ax52.plot(result.y[:, ii*10], zN, '-')
    if 30<=t<225:
        ax53.plot(result.y[:, ii*10], zN, '-')
ax51.set_title('0-25d')
ax52.set_title('25-30d')
ax53.set_title('30-225d')
fig5.suptitle('Water head (m) vs. Depth (m)')
ax51.grid(b=True)
ax52.grid(b=True)
ax53.grid(b=True)

fig6, ax6 = plt.subplots(figsize=(4, 7))
for ii in np.arange(0, nOut, 10):
    y = ThetaOut[:,ii].reshape(100,1)
    ax6.plot(y, zN, '-')
ax6.set_xlabel('Water Content')
ax6.set_xlabel('Depth (m)')
fig6.suptitle('Water Content vs. Depth (m)')


fig7, (ax71,ax72,ax73) = plt.subplots(1,3,figsize=(8, 6))
ii = 0
for ii,t in enumerate((result.t[0:len(result.t):10]).tolist()):
    if t<25:
        y = ThetaOut[:,ii*10].reshape(100,1)
        ax71.plot(y, zN, '-')
        
    if 25<=t<30:
        y = ThetaOut[:, ii*10].reshape(100,1)
        ax72.plot(y, zN, '-')
    if 30<=t<225:
        y = ThetaOut[:, ii*10].reshape(100,1)
        ax73.plot(y, zN, '-')
ax71.set_title('0-25d')
ax72.set_title('25-30d')
ax73.set_title('30-225d')
fig7.suptitle('Water content vs. Depth (m)')
ax71.grid(b=True)
ax72.grid(b=True)
ax73.grid(b=True)

plt.show()
 

