import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.signal import find_peaks 
import uncertainties.unumpy as unp 
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline

plt.rc('font', size=14, weight='normal')
#mpl.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['legend.fontsize'] = 14
plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14) 


#Auslesen der Daten ACHTUNG: Daten müssen aus Winkel ins Bogenmaßumgerechnet werden und für die Plots wieder zurück

alpha_det, I_det = np.genfromtxt("Detektorscan.UXD", unpack = True)

z1, I_z1 = np.genfromtxt("zScan_1.UXD", unpack = True)

rocking_0, I_rocking0 = np.genfromtxt("Rockingscan_1.UXD", unpack = True)
rocking_0 = rocking_0*np.pi/180

rocking_03, I_rocking03 = np.genfromtxt("Rockingscan_2.UXD", unpack = True)
rocking_03 = rocking_03*np.pi/180

z3, I_z3 = np.genfromtxt("zScan_3.UXD", unpack = True)

theta2, I_theta2 = np.genfromtxt("Messung.UXD", unpack = True)
#theta2 = theta2*np.pi/180

theta2_diffus, I_diffus = np.genfromtxt("Diffusion.UXD", unpack = True)
theta2_diffus = theta2_diffus*np.pi/180


def gauß(x, sigma, bg, x0, A):
    return A*np.exp(-(x-x0)**2/(2*sigma**2)) #+ bg #/(np.sqrt(2*np.pi) * sigma)




#1 Gauß an Intensitätskurve anpassen und daraus maximale Intensität und Halbwertsbreite bestimmen
def f(x, const):
    return const

guess =([0.2, 40, 0, 900000])

params_gauß, cov_gauß = curve_fit(gauß, alpha_det, I_det, p0=guess)
x = ([-params_gauß[0]/2, params_gauß[0]/2])
y = ([905560.534467708/2, 905560.534467708/2])
alpha_fit = np.linspace(-0.5, 0.5, 1000)

fit = gauß(alpha_fit, *params_gauß)
spline = UnivariateSpline(alpha_fit, fit-np.max(fit)/2, s=0)
r1, r2 = spline.roots() # find the roots


fig, axs = plt.subplots(figsize=(7,6))
axs.plot(alpha_det, I_det, "x", label="Intensitätsverteilung")
axs.axvspan(r1, r2, facecolor='grey', alpha=0.3, label=r"FWHM=0,1062°")
axs.set_xlabel("Alpha [°]")
axs.set_ylabel("I [will. Ein.]")
axs.plot(alpha_fit, gauß(alpha_fit, *params_gauß), label="Gauß-Anpassung")
axs.legend(loc="best")
plt.savefig("gauß.pdf")

I_0 = I_det.max()


print(f"""Halbwertsbreite: {r2-r1} oder über stddev: {params_gauß[0]*2*np.sqrt(2*np.log(2))}
Maximum bei {params_gauß[3]+params_gauß[1]}""")


#2 Reflektivität minus diefussen Untergrund in Diagramm auftragen

def R_fresnel(alpha_i, alpha_c, lam, mu):
    return ((alpha_i - np.sqrt(0.5 * (np.sqrt((alpha_i**2-alpha_c**2)**2 + 4*(lam*mu/(4*np.pi))**2 ) + (alpha_i**2-alpha_c**2))))**2 + (0.5 * (np.sqrt((alpha_i**2-alpha_c**2)**2 + 4*(lam*mu/(4*np.pi))**2 ) - (alpha_i**2-alpha_c**2))))/((alpha_i + np.sqrt(0.5 * (np.sqrt((alpha_i**2-alpha_c**2)**2 + 4*(lam*mu/(4*np.pi))**2 ) + (alpha_i**2-alpha_c**2))))**2 + (0.5 * (np.sqrt((alpha_i**2-alpha_c**2)**2 + 4*(lam*mu/(4*np.pi))**2 ) - (alpha_i**2-alpha_c**2))))

def R_approx(alpha_i, alpha_c):
    return (alpha_c/(2*alpha_i))**4

reflectivity = (I_theta2 - I_diffus)/(I_0*5) #Hintergrund abziehen
theta_fit = np.linspace(0, 1.3, 1000)

alpha_c=0.223

fig, axs = plt.subplots(figsize=(7,6))
axs.plot(theta2[0:261], reflectivity[0:261], "-", label="Reflektivität")
axs.set_xlabel(r"$\alpha_i$ [°]")
axs.set_ylabel(r"$I/I_0$ [will. Ein.]")
axs.plot(theta_fit, R_fresnel(theta_fit *np.pi/180, 0.223*np.pi/180, 1.54*10**(-10), 14100), label="$R_F$")
axs.set_yscale('log')
axs.legend(loc="best")
plt.savefig("reflectivity.pdf")




#3 Geometriefaktor G aus erstem Rocking Scan bestimmen und Daten korrigieren.

#Strahlbreite aus halber Abschattung des ersten z-Scans bestimmen

df = np.diff(I_z1)/np.diff(z1)

fig, axs = plt.subplots(figsize=(7,6))
axs.plot(z1, I_z1/(I_z1.max()), "-", label="Intensität")
axs.plot(z1[1:], df/((np.abs(df)).max()), "-", label="Diff.")
axs.axvspan(0.04, 0.26, facecolor='grey', alpha=0.3, label=r"$d_0=22$mm")
axs.set_xlabel("z-Position [will. Ein.]")
axs.set_ylabel(r"$I/I_0$ [will. Ein.]")
#axs.plot(alpha_fit, gauß(alpha_fit, *params_gauß), label="Gauß-Anpassung")
axs.legend(loc="best")
plt.savefig("abschattung.pdf")

d_0=2*(0.15-0.04) # Strahlbreite in mm
print(f"""halbe Strahlbreite: {0.15-0.04}mm und ganze Strahlbreite: {d_0}mm""")


D=20 #Probenlänge in mm

alpha_g = np.arcsin(d_0/D) #Aus Strahl- und Probenbreite bestimmter Geometriewinkel im Bogenmaß


fig, axs = plt.subplots(figsize=(7,6))
axs.plot(rocking_0*180/(np.pi), I_rocking0/(I_rocking0.max()), "-", label="Intensität")
axs.vlines(-0.65,0, 1, alpha=0.3, label=r"$\alpha_{g1}=0,65$°")
axs.vlines(0.63,0, 1, alpha=0.3, label=r"$\alpha_{g2}=0,63$°")
axs.set_xlabel(r"$\alpha$ [°]")
axs.set_ylabel(r"$I/I_0$ [will. Ein.]")
axs.legend(loc="best")
plt.savefig("rockingscan_1.pdf")

print(f"""Geometriewinkel aus Strahl- und Probenbreite: {alpha_g*180/(np.pi)}°
Geometriewinkel als Mittelwert aus Rockingscan: {(0.65+0.63)/2}°""")

#Reflektivitätsscan mit korrigierten Daten und daraus abgeschätzte Schichtdicke

reflectivity_cor = np.append(reflectivity[0:46], reflectivity[46:]*D*np.sin(theta2[46:] *np.pi/180)/d_0)

minima, minima_props = find_peaks(-reflectivity_cor[:261], distance=7)
minima=minima[1:-2]
print(minima)
#print(theta2[24,33,42,52,62,72,82,87,93,103,114,125,128,135,145,147,149,151,155,157,161,167,169,171,173,177,179,182,187,191,196,200,208,211,213])

fig, axs = plt.subplots(figsize=(7,6))
axs.plot(theta2[0:261], reflectivity[0:261], "-", label="Reflektivität")
axs.plot(theta2[0:261], reflectivity_cor[0:261], "-", label="korr. Reflektivität")
axs.plot(theta2[minima], reflectivity_cor[minima], "x", label="Minima")
axs.set_xlabel(r"$\alpha_i$ [°]")
axs.set_ylabel(r"$I/I_0$ [will. Ein.]")
axs.plot(theta_fit, R_fresnel(theta_fit *np.pi/180, 0.223*np.pi/180, 1.54*10**(-10), 14100), label="$R_F$")
axs.set_yscale('log')
axs.legend(loc="best")
plt.savefig("reflectivity_cor.pdf")
