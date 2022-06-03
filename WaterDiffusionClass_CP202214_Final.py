
# coding: utf-8
"""
# # This code originally was "HeatFlow Model"created by @author: theimovaara 
on Sun May  9 13:46:11 2021

Then Modified into Water Flux Model (Richard Equation) by Qroup CP2022 14 
for the completion of Assignment 2 _CIE4365-16


# First we need to import the packages which we will be using:

"""
# In[1]:

import numpy as np
import pandas as pd
import scipy.integrate as spi
import scipy.sparse as sp
import MyTicToc as mt
import matplotlib.pyplot as plt
import seaborn as sns


class WaterDiffusion:
# ## Definition of functions
# Then we need to define the functions which we will be using:
# BndTTop for calculating the top temperature as a function of time;
# HeatFlux for calculating all heat fluxes in the domain;
# DivHeatFlux for calculating the divergence of the heat flux across the cells in the domain.
    def __init__(self, sPar, mDim, bPar):
        self.sPar = sPar
        self.mDim = mDim
        self.bPar = bPar
    
    def SeffFun(self, hw):#Se = Effective Saturation = Seff
        hc = -hw
        Seff = (1 + ((hc * (hc > 0)) * self.sPar.vGA) ** self.sPar.vGN) ** (-self.sPar.vGM)
        return Seff
        
    ##FromL3_Slide13, Theta is determined in term of Effective Saturation
    def ThetaFun (self, hw):
        Seff = self.SeffFun(hw)
        Theta = self.sPar.thetaRes + Seff * (self.sPar.thetaSat - self.sPar.thetaRes)
        return Theta
    
    ##Determining the C using the complex function presented in Imaginary Derivative article
    def CComplxFun(self, hw):
        dhw = np.sqrt(np.finfo(float).eps) 
        Theta = self.ThetaFun(hw.real + 1j * dhw)
        C = Theta.imag / dhw
        return C
    
    ##Determining the C_Prime = Chw (Calculation) + Sw (Formula:FromL3_Slide13) * Ssw (Formula: FromL3_Slide22/29)
    def CPrimeFun (self, hw):
        #1. Determine volumatric Water content
        Theta = self.ThetaFun(hw)
        #2. Determine Water Saturation 
        Sw = Theta / self.sPar.thetaSat  
        #3. Determine the differential water capacity: Chw
        Chw = self.CComplxFun(hw)
        
        
        #4: Constants: gConstant = 9.81 [m/s2] | density of water: roW = 1000 [kg/m3] | Compressibility of water: betaW= 4.5e-10 [1/Pa]
        rho = 1000 #[kg/m3]
        g = 9.81 #[m/s2]
        betaW = 4.5e-10 #[1/Pa]
        #5: Calculate Ssw
        Ssw = rho * g * (self.sPar.Cv + self.sPar.thetaSat * betaW)
        #6. Calculate C_Prime
        C_Prime = Chw + Sw * Ssw
        
        #Matrix Dimention
        nN = self.mDim.nN
        nIN = self.mDim.nIN
        C_Prime[nN-1] = 1/self.mDim.dzIN[nIN-2] * (hw[nN-1]>0) + C_Prime[nN-1]\
            * (hw[nN-1]<=0)
        return C_Prime
    
    def KFun(self, hw):
        #KFun = Ksat * Krw
        nr,nc = hw.shape
        nIN = self.mDim.nIN
        Seff = self.SeffFun(hw)
        K = self.sPar.KSat * Seff ** 3
        
        KIN = np.zeros([nIN,nc], dtype=hw.dtype)
        KIN[0] = K[0]
        ii = np.arange(1, nIN - 1)
        KIN[ii] = np.minimum(K[ii - 1], K[ii])
        KIN[nIN - 1] = K[nIN - 2]
        return KIN

    
    def WaterFlux(self, t, hw, nOut):
        nr,nc = hw.shape
        nIN = self.mDim.nIN
        dzN = self.mDim.dzN   
        qw = np.zeros((nIN, nc), dtype=hw.dtype)


        KIN = self.KFun(hw)

        
        if self.bPar.WatBotCond == 'gravity':  #bottom boundary condition
            qw[0] = -KIN[0]
        else:
            qw[0] = -self.bPar.BotRobResTerm * (hw[0] - self.bPar.externalrobhead)                  #if not gravity, Robin


        # Flux at all internodes
        ii = np.arange(1, nIN-1)  #excluding last element
        qw[ii] = -KIN[ii] * ((hw[ii] - hw[ii-1]) / dzN[ii-1] + 1)
        
        return qw
    
    ## Divergent of the Water Flux
    def DivWaterFlux(self, t, hw, nOut):
        nr,nc = hw.shape
        nN = self.mDim.nN
        dzIN = self.mDim.dzIN
        C_Prime = self.CPrimeFun(hw)
        
       # Calculate Water fluxes accross all internodes
        qWat = self.WaterFlux(t, hw, nOut)
        divqWat = np.zeros([nN, nc],dtype=hw.dtype)
        
        # Calculate divergence of flux for all nodes
        i = np.arange(0, nN)
        divqWat[i] = -((qWat[i + 1] - qWat[i])/ (dzIN[i] * C_Prime[i]))
        
        return divqWat
    
 
    
        
    def IntegratehwF(self, t, hwIni, nOut):

        def dYdt(t, hw):
            if len(hw.shape)==1:
                hw = hw.reshape(self.mDim.nN,1)
            rates = self.DivWaterFlux(t, hw, nOut)
            return rates
            
        def jacFun(t, y):
            if len(y.shape)==1:
                y = y.reshape(self.mDim.nN,1)
            
            nr, nc = y.shape
            dh = np.sqrt(np.finfo(float).eps)
            ycmplx = y.copy().astype(complex)
            ycmplx = np.repeat(ycmplx,nr,axis=1)
            c_ex = np.ones([nr,1])* 1j*dh
            ycmplx = ycmplx + np.diagflat(c_ex,0)
            dfdy = dYdt(t, ycmplx).imag/dh
            return sp.coo_matrix(dfdy)
            
        # solve rate equation
        t_span = [t[0],t[-1]]
        int_result = spi.solve_ivp(dYdt, t_span, hwIni.squeeze(), 
                                   t_eval=t, 
                                   method='BDF', vectorized=True, jac=jacFun,  
                                   rtol=1e-8)
        return int_result





