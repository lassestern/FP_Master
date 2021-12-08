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

def hochdrei(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

def mean(x):
    return np.sum(x)/(len(x))


#####Anpassung von B(I), um Kalibrierung zu erhalten


I, B = np.genfromtxt("data/magnetfeld.dat", unpack = "True") #I in Ampere, B in mT

B = B*0.001 #B in Tesla

params, cov_params = curve_fit(hochdrei, I, B)

print(I, B)


x = np.linspace(0, 7.2)


fig, axs = plt.subplots(figsize=(7,6))
axs.plot(I, B*1000, "x", label="B-Feld")
axs.set_xlabel("I [A]")
axs.set_ylabel("B [mT]")
axs.plot(x, hochdrei(x, *params)*1000, label="Fit")
axs.legend(loc="best")
plt.savefig("magnetfeld.pdf")

print(f"""Fitparameter: {params}""")


######Daten der Aufspaltungen auslesen
l_Abstand_rot_sigma, aufspaltung_rot_sigma = np.genfromtxt("data/rot_sigma.dat", unpack = "True") #Abstände in Pixeln 
l_Abstand_rot_sigma = unp.uarray(l_Abstand_rot_sigma, 1)
aufspaltung_rot_sigma = unp.uarray(aufspaltung_rot_sigma, 1)

l_Abstand_blau_sigma, aufspaltung_blau_sigma = np.genfromtxt("data/blau_sigma.dat", unpack = "True") #Abstände in Pixeln 
l_Abstand_blau_sigma = unp.uarray(l_Abstand_blau_sigma, 1)
aufspaltung_blau_sigma = unp.uarray(aufspaltung_blau_sigma, 1)

l_Abstand_blau_pi, aufspaltung_blau_pi = np.genfromtxt("data/blau_pi.dat", unpack = "True") #Abstände in Pixeln 
l_Abstand_blau_pi = unp.uarray(l_Abstand_blau_pi, 1)
aufspaltung_blau_pi = unp.uarray(aufspaltung_blau_pi, 1)



#Zunächst die Dispersionsgebiete berechnen
def disp(lam, d, n):
    return lam**2/(2*d*np.sqrt(n**2 - 1))

d = 0.004 #Dicke der Platte in Meter
#rotes Licht: λ=643,8 nm
lam_rot=643.8 * 10**(-9) #Wellenlänge in Meter
n_rot = 1.4567
disp_rot = disp(lam_rot, d, n_rot)
print(f"""Dispersionsgebiet rotes Licht: {disp_rot}""")

#blaues Licht: λ=480 nm
n_blau = 1.4635
lam_blau=480 * 10**(-9) #Wellenlänge in Meter
disp_blau = disp(lam_blau, d, n_blau)
print(f"""Dispersionsgebiet blaues Licht: {disp_blau}""")


##Dann die Wellenlängenaufspaltung berechnen

def delta_lam(aufspaltung, abstand, disp):
    return aufspaltung * disp/(2*abstand) 

delta_lam_rot_sigma = delta_lam(aufspaltung_rot_sigma, l_Abstand_rot_sigma, disp_rot) 
delta_lam_blau_sigma = delta_lam(aufspaltung_blau_sigma, l_Abstand_blau_sigma, disp_blau)
delta_lam_blau_pi = delta_lam(aufspaltung_blau_pi, l_Abstand_blau_pi, disp_blau)


print(f"""
Aufspaltung rot sigma: {delta_lam_rot_sigma}
Aufspaltung blau sigma: {delta_lam_blau_sigma}
Aufspaltung blau pi: {delta_lam_blau_pi}
""")

#Die angelegten Magnetfelder aus der Stromstärke bestimmen

B_rot_sigma = hochdrei(5, *params)
B_blau_sigma = hochdrei(2.6, *params)
B_blau_pi = 1.009

print(f"""
B rot sigma: {B_rot_sigma}
B blau sigma: {B_blau_sigma}
B blau pi: {B_blau_pi}
""")

#Zuletzt die Lande-Faktoren aus den Mittelwerten der Aufspaltung berechnen

delta_lam_rot_sigma = mean(delta_lam_rot_sigma)
delta_lam_blau_sigma = mean(delta_lam_blau_sigma)
delta_lam_blau_pi = mean(delta_lam_blau_pi)

print(f"""
Aufspaltung rot sigma: {delta_lam_rot_sigma}
Aufspaltung blau sigma: {delta_lam_blau_sigma}
Aufspaltung blau pi: {delta_lam_blau_pi}
""")



def lande(delta_lam, B, lam):
    return delta_lam * const.h * const.c/(const.value(u'Bohr magneton')*B*lam**2)

lande_rot_sigma = lande(delta_lam_rot_sigma, B_rot_sigma, 643.8*10**(-9))
a_lande_rot_sigma = (lande_rot_sigma - 1)/(1)
lande_blau_sigma = lande(delta_lam_blau_sigma, B_blau_sigma, 480*10**(-9))
a_lande_blau_sigma = (lande_blau_sigma - 1.75)/1.75
lande_blau_pi = lande(delta_lam_blau_pi, B_blau_pi, 480*10**(-9))
a_lande_blau_pi = (lande_blau_pi - 0.5)/0.5


print(f"""
Lande-Faktor rot sigma: {lande_rot_sigma}
Abweichung zur Theorie: {a_lande_rot_sigma}
Lande-Faktor blau sigma: {lande_blau_sigma}
Abweichung zur Theorie: {a_lande_blau_sigma}
Lande-Faktor blau pi: {lande_blau_pi}
Abweichung zur Theorie: {a_lande_blau_pi}
""")


#a = ((n_norm-1) - (n_norm_lit-1))/(n_norm_lit-1)
#print(f"Theoriewert: {n_norm_lit}")
#print(f"Abweichung zum Theoriewert: {a}")


