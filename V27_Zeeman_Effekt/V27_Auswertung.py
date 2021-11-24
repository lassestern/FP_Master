#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.signal import find_peaks 
import uncertainties.unumpy as unp 
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
from scipy.optimize import curve_fit

plt.rc('font', size=14, weight='normal')
#mpl.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['legend.fontsize'] = 14
plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14) 



#Formel für die Standardabweichung des Mittelwerts
def stanni(Zahlen, Mittelwert):
    i=0
    s=0
    n=len(Zahlen)
    while i<len(Zahlen):
        s = s + (Zahlen[i] - Mittelwert)**2
        i = i + 1
    return np.sqrt(s/(n*(n-1)))

def linear(x, m, b):
    return m*x+b

I, B = np.genfromtxt("data/magnetfeld.dat", unpack = "True") #I in Ampere, B in mT

B = B*0.001 #B in Tesla

params, cov_params = curve_fit(linear, I, B)

print(I, B)


x = np.linspace(0, 7.2)


fig, axs = plt.subplots(figsize=(7,6))
axs.plot(I, B*1000, "x", label="B-Feld")
axs.set_xlabel("I [A]")
axs.set_ylabel("B [mT]")
axs.plot(x, linear(x, *params)*1000, label="Fit")
axs.legend(loc="best")
plt.savefig("magnetfeld.pdf")


#
#
#n_norm = unp.sqrt(1+ 1.013*295.05*n_quad_params_unc[0]/288.15)
#print("Brechungsindex von Luft bei Normalbedingungen")
#print(n_norm)
#
#n_norm_lit = 1.00027653
#
#a = ((n_norm-1) - (n_norm_lit-1))/(n_norm_lit-1)
#print(f"Theoriewert: {n_norm_lit}")
#print(f"Abweichung zum Theoriewert: {a}")


