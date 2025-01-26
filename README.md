# Fischer-Tropsch-Reactor-Design-Code
This code is written to perform design calculations of fixed bed reactor which can be used in Fischer Tropsch Process for gasoline type hydrocarbon synthesis
Reactor_type = "Non-Isothermal"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from math import log, log10, exp, e, pow, sqrt
from scipy.integrate import odeint
Tfluid = 245
Tl = 280
pi = 3.141592
Tube_number = 1
Ws = 84
D = 0.079 # m
fi = 0.6 # Void Fraction
Areac = (D**2)*(pi)/4
Wtotal = Ws*Tube_number
Ftotal_initial = 11
Ftotal_flow = Ftotal_initial*Tube_number # mol/s
Xco0 = 21.4/100 # 21.4
Xco20 = 35.4/100
Xh20 = 36.5/100 # 36.5
Xh2o0 = 0.7/100
Xc1i = 5.6/100
Xn20 = 0.4/100
Fco0 = Ftotal_initial*Xco0
Fco20 = Ftotal_initial*Xco20
Fh20 = Ftotal_initial*Xh20
Fh2o0 = Ftotal_initial*Xh2o0
Fc1i = Ftotal_initial*Xc1i
Fn20 = Ftotal_initial*Xn20
R = 8.314
Pxi = 17.5
Ptotal0 = Pxi*(10**5)
T0 = Tl+273
H_paraffin = -170 # kJ/mol
H_oleffin = -165 # kJ/mol
H_methane = -205 # kJ/mol
H_WGS = -41 # kJ/mol
U = 300 # W/m2.K
Ta = Tfluid + 273 # K
Cp = 3.5*R
Ma = 23.45 #g/mol
pg0 = ((Ptotal0/(R*T0))*Ma)/1000 # Initial gas density kg/m^3  # 
pCat = 7900 # kg/m^3 
Dp = 7.4/1000 # Pellet diameter in meters
Areac = (D**2)*(pi)/4
Total_mass = Fco0*28 + Fco20*44 + Fh2o0*18 + Fh20*2 + Fn20*28 + Fc1i*16 # g 
G = (Total_mass/Areac)/1000 # kg/m2.s
mu = 23.13E-7 # Viscosity Pa.s
gc = 1
Laminar_term = (150*(1-fi)*mu)/(Dp)
Turbulent_term = 1.75*G
Cterm = Laminar_term + Turbulent_term
Bterm = (1-fi)/((fi)**3)
Aterm = G/(pg0*Dp*gc)
beta0 = (Aterm)*(Bterm)*(Cterm)
L = Ws/(Areac*(1-fi)*pCat)
def fixed_bed_reactor(F,Wx):
    Fco = F[0]
    Fh2o = F[1]
    Fco2 = F[2]
    Fh2 = F[3]
    Fc1 = F[4]
    Fc2 = F[5]
    Fc3 = F[6]
    Fc4 = F[7]
    Fc5 = F[8]
    Fc6 = F[9]
    Fc7 = F[10]
    Fc8 = F[11]
    Fc9 = F[12]
    Fc10 = F[13]
    Fn2 = F[14]
    #Oleffins
    Fcc2 = F[15]
    Fcc3 = F[16]
    Fcc4 = F[17]
    Fcc5 = F[18]
    Fcc6 = F[19]
    Fcc7 = F[20]
    Fcc8 = F[21]
    Fcc9 = F[22]
    Fcc10 = F[23]
    T = F[24]
    Ptotal = F[25]
    Ftotal = Fco + Fh2o + Fco2 + Fh2 + Fc1 + Fc2 + Fc3 + Fc4 + Fc5 + Fc6 + Fc7 + Fc8 + Fc9 + Fc10 + Fn2 + Fcc2 + Fcc3 + Fcc4 + Fcc5 + Fcc6 + Fcc7 + Fcc8 + Fcc9 + Fcc10  
    R = 8.314 # J/mol.K # K
    Pco = Ptotal*(Fco/Ftotal)
    Ph2 = Ptotal*(Fh2/Ftotal)
    Ph2o = Ptotal*(Fh2o/Ftotal)
    Pco2 = Ptotal*(Fco2/Ftotal)
    # Rate Constants and Rate Expressions
    kHC1 = 1.22*(10**(-10))*1000 # mol/(kg.s.Pa)
    E5 = 94.5 # kJ/mol
    kHC5 = (4.326E-3)*exp(-E5*1000/(R*T))*1000
    E6 = 132.3 # kJ/mol
    kHC6 = (2.71E6)*exp(-E6*1000/(R*T))*1000
    alpha = (kHC1*Pco)/((kHC1*Pco)+(kHC5*Ph2)+(kHC6))
    term1 = (kHC1*Pco)/((kHC1*Pco)+(kHC5*Ph2))
    rp1 = ((kHC5*Ph2)*term1)/(1+(term1*(1/(1-alpha))))
    rp2 = ((kHC5*Ph2)*term1)*(alpha)/(1+(term1*(1/(1-alpha))))
    rp3 = ((kHC5*Ph2)*term1)*(alpha**2)/(1+(term1*(1/(1-alpha))))
    rp4 = ((kHC5*Ph2)*term1)*(alpha**3)/(1+(term1*(1/(1-alpha))))
    rp5 = ((kHC5*Ph2)*term1)*(alpha**4)/(1+(term1*(1/(1-alpha))))
    rp6 = ((kHC5*Ph2)*term1)*(alpha**5)/(1+(term1*(1/(1-alpha))))
    rp7 = ((kHC5*Ph2)*term1)*(alpha**6)/(1+(term1*(1/(1-alpha))))
    rp8 = ((kHC5*Ph2)*term1)*(alpha**7)/(1+(term1*(1/(1-alpha))))
    rp9 = ((kHC5*Ph2)*term1)*(alpha**8)/(1+(term1*(1/(1-alpha))))
    rp10 = ((kHC5*Ph2)*term1)*(alpha**9)/(1+(term1*(1/(1-alpha))))
    ro2 = ((kHC6)*term1)*(alpha**1)/(1+(term1*(1/(1-alpha))))
    ro3 = ((kHC6)*term1)*(alpha**2)/(1+(term1*(1/(1-alpha))))
    ro4 = ((kHC6)*term1)*(alpha**3)/(1+(term1*(1/(1-alpha))))
    ro5 = ((kHC6)*term1)*(alpha**4)/(1+(term1*(1/(1-alpha))))
    ro6 = ((kHC6)*term1)*(alpha**5)/(1+(term1*(1/(1-alpha))))
    ro7 = ((kHC6)*term1)*(alpha**6)/(1+(term1*(1/(1-alpha))))
    ro8 = ((kHC6)*term1)*(alpha**7)/(1+(term1*(1/(1-alpha))))
    ro9 = ((kHC6)*term1)*(alpha**8)/(1+(term1*(1/(1-alpha))))
    ro10 = ((kHC6)*term1)*(alpha**9)/(1+(term1*(1/(1-alpha))))
    # KewG Calculation Start
    j1 = 5078.0045/T
    j2 = -5.897
    j3 = (13.95E-4)*T
    j4 = -(27.59E-8)*(T**2)
    Kewg = exp(j1+j2+j3+j4)
    # Kewg Calculation Finish
    Ev = 27.7 # kJ/mol
    kv = (5.90E-10)*exp(-Ev*1000/(R*T))*1000 # kg
    Kv = 3.6*(10**(-2.5)) # Pa^-0.5
    # CO2 Rate Start
    u1 = Pco*Ph2o/(sqrt(Ph2))
    u2 = ((1/Kewg)*Pco2)*(sqrt(Ph2))
    u3 = kv*(u1-u2) # Upper Term
    u4 = (1 +(Kv*Ph2o/(sqrt(Ph2))))**2
    Rco2 = u3/u4 
    # CO2 Rate End
    rC1 = rp1
    rC2 = rp2
    rC3 = rp3
    rC4 = rp4
    rC5 = rp5
    rC6 = rp6
    rC7 = rp7
    rC8 = rp8
    rC9 = rp9
    rC10 = rp10
    rCC2 = ro2
    rCC3 = ro3
    rCC4 = ro4
    rCC5 = ro5
    rCC6 = ro6
    rCC7 = ro7
    rCC8 = ro8
    rCC9 = ro9
    rCC10 = ro10
    # Carbon monoxide Consumption Start
    rCo_paraffin = rC1 + 2*rC2 + 3*rC3 + 4*rC4 + 5*rC5 + 6*rC6 + 7*rC7 + 8*rC8 + 9*rC9 + 10*rC10 
    rCo_oleffin = 2*ro2 + 3*ro3 + 4*ro4 + 5*ro5 + 6*ro6 + 7*ro7 + 8*ro8 + 9*ro9 + 10*ro10 
    # Carbon Monoxide Consumption End
    # H2 Consumption Start
    rH2_paraffin = 3*rC1 + 5*rC2 + 7*rC3 + 9*rC4 + 11*rC5 + 13*rC6 + 15*rC7 + 17*rC8 + 19*rC9 + 21*rC10 
    rH2_oleffin = 2*2*ro2 + 2*3*ro3 + 2*4*ro4 + 2*5*ro5 + 2*6*ro6 + 2*7*ro7 + 2*8*ro8 + 2*9*ro9 + 2*10*ro10
    # H2 Consumption End
    # H2O Formation Start
    rH2O_paraffin = rCo_paraffin
    rH2O_oleffin  = rCo_oleffin
    # Enthalpy
    dH1 = rC1*(1000*H_methane) + (rC2 + rC3 + rC4 + rC5 + rC6 + rC7 + rC8 + rC9 + rC10)*(1000*H_paraffin) + (rCC2 + rCC3 + rCC4 + rCC5 + rCC6 + rCC7 + rCC8 + rCC9 + rCC10)*(1000*H_oleffin) + Rco2*(H_WGS)
    dH = -dH1
    # Differential Equations
    dcodz = -rCo_paraffin - rCo_oleffin - Rco2
    dh2odz = rH2O_paraffin + rH2O_oleffin - Rco2
    dco2dz = Rco2
    dh2dz = -rH2_paraffin - rH2_oleffin + Rco2
    dc1dz = rC1
    dc2dz = rC2
    dc3dz = rC3
    dc4dz = rC4
    dc5dz = rC5
    dc6dz = rC6
    dc7dz = rC7
    dc8dz = rC8
    dc9dz = rC9
    dc10dz = rC10
    dn2dw = 0
    do2dw = rCC2
    do3dw = rCC3
    do4dw = rCC4
    do5dw = rCC5
    do6dw = rCC6
    do7dw = rCC7
    do8dw = rCC8
    do9dw = rCC9
    do10dw = rCC10
    dtdw = (((U*(4/D))*(Ta-T)/(pCat*(1-fi)))+dH)/(Ftotal*Cp)
    dpdw = -((beta0)/(Areac*(1-fi)*pCat))*(Ptotal0/Ptotal)*(T/T0)*(Ftotal/Ftotal_initial)
    output = dcodz,dh2odz,dco2dz,dh2dz,dc1dz,dc2dz,dc3dz,dc4dz,dc5dz,dc6dz,dc7dz,dc8dz,dc9dz,dc10dz,dn2dw,do2dw,do3dw,do4dw,do5dw,do6dw,do7dw,do8dw,do9dw,do10dw,dtdw,dpdw
    return output
Wx = np.linspace(0,Ws,1000) # kg
F0 = [Fco0,Fh2o0,Fco20,Fh20,Fc1i,0,0,0,0,0,0,0,0,0,Fn20,0,0,0,0,0,0,0,0,0,T0,Ptotal0]
F = odeint(fixed_bed_reactor, F0,Wx)
W = Wx/1 # kg
Fco = F[:,0]
Fh2o = F[:,1]
Fco2 = F[:,2]
Fh2 = F[:,3]
Fc1 = F[:,4]
Fc2 = F[:,5]
Fc3 = F[:,6]
Fc4 = F[:,7]
Fc5 = F[:,8]
Fc6 = F[:,9]
Fc7 = F[:,10]
Fc8 = F[:,11]
Fc9 = F[:,12]
Fc10 = F[:,13]
Fn2 = F[:,14]
Fcc2 = F[:,15]
Fcc3 = F[:,16]
Fcc4 = F[:,17]
Fcc5 = F[:,18]
Fcc6 = F[:,19]
Fcc7 = F[:,20]
Fcc8 = F[:,21]
Fcc9 = F[:,22]
Fcc10 = F[:,23]
T = F[:,24]
Ptotal = F[:,25]
Ftotalx = Fc1 + Fc2 + Fc3 + Fc4 + Fc5 + Fc6 + Fc7 + Fc8 + Fc9 + Fc10  + Fcc2 + Fcc3 + Fcc4 + Fcc5 + Fcc6 + Fcc7 + Fcc8 + Fcc9 + Fcc10 + Fco + Fco2 + Fh2 + Fn2 + Fh2o
Fdesired = Fc5 + Fc6 + Fc7 + Fc8 + Fc9 + Fc10 + Fcc5 + Fcc6 + Fcc7 + Fcc8 + Fcc9 + Fcc10
Pco = (Ptotal*Fco)/(Ftotalx)
Ph2 = (Ptotal*Fh2)/(Ftotalx)
Ph2o = Ptotal*(Fh2o/Ftotalx)
Pco2 = Ptotal*(Fco2/Ftotalx)
kHC1 = 1.22*(10**(-10))*1000# mol/(kg.s.Pa)
E5 = 94.5 # kJ/mol
kHC5 = (4.326E-3)*np.exp(-E5*1000/(R*T))*1000
E6 = 132.3 # kJ/mol
kHC6 = (2.71E6)*np.exp(-E6*1000/(R*T))*1000
alpha = (kHC1*Pco)/((kHC1*Pco)+(kHC5*Ph2)+(kHC6))
R = 8.314
fig = plt.figure(figsize=(10,7))
plt.plot(W,Fh2,label="H2")  
plt.plot(W,Fco,label="CO")
plt.plot(W,Fc1,label="C1")
plt.plot(W,Fc2,label="C2")
plt.plot(W,Fc3,label="C3")
plt.plot(W,Fc4,label="C4")
plt.plot(W,Fc5,label="C5")
plt.plot(W,Fc6,label="C6")
plt.plot(W,Fc7,label="C7")
plt.plot(W,Fc8,label="C8")
plt.plot(W,Fc9,label="C9")
plt.plot(W,Fc10,label="C10")
plt.plot(W,Fh2o,color="black",label="H2O")
plt.plot(W,Fn2,label="N2")
plt.plot(W,Fco2,label="CO2")
plt.plot(W,Fcc2,label="C2 olef")
plt.plot(W,Fcc3,label="C3 olef")
plt.plot(W,Fcc4,label="C4 olef")
plt.plot(W,Fcc5,label="C5 olef")
plt.plot(W,Fcc6,label="C6 olef")
plt.plot(W,Fcc7,label="C7 olef")
plt.plot(W,Fcc8,label="C8 olef")
plt.legend(fontsize=10)
plt.title(Reactor_type+" Reactor Product Stream Gas Compostion")
plt.xlabel("W Catalyst Weight (kg)")
plt.ylabel("Gas Flowrate (mol/s)")
plt.grid()
Selectivity = (Fdesired)/(Ftotalx)
Mdesired = (Fc5*72 + Fc6*86 + Fc7*100 + Fc8*114 + Fc9*128 + Fc10*142 + Fcc5*70 + Fcc6*84 + Fcc7*98 + Fcc8*112 + Fcc9*126 + Fcc10*140)/(Fc5*72 + Fc6*86 + Fc7*100 + Fc8*114 + Fc9*128 + Fc10*142 + Fcc5*70 + Fcc6*84 + Fcc7*98 + Fcc8*112 + Fcc9*126 + Fcc10*140 + Fc1*16 + Fc2*30 + Fc3*44 + Fc4*58 + Fcc2*28 + Fcc3*42 + Fcc4*56)*100
Mass_s = Mdesired[-1]
Conversion =((Fco0-Fco[-1])/Fco0)*100
# Results
print("Initial Temperature = "+str(Tl)+" C")
print("Final Temperature = "+str(round(T[-1]-273,2))+" C")
print("Initial Pressure = "+str(Pxi)+" bar")
print("Final Pressure = "+str(round((Ptotal[-1]/100000),2))+" bar")
print("CO Conversion "+ str(round(Conversion,3))+"%")
print("Selectivity "+str(round(Mass_s,3))+" %" )
plist = ["Initial Temperature (C)","Final Temperature (C)","Initial Pressure (bar)","Final Pressure (bar)","CO Conversion %",
        "Selectivity %","Annual production (ton/year)","Tube Number","Pipe Diameter (cm)","Required Pipe Length (m)"
        ,"Pellet Diameter (mm)","Coolant Temperature (C)","Catalyst Density (kg/m^3)","Void Fraction","Total Catalyst Weight (ton)","Feed per tube (mol/s)"]
F_carbon_initial=Fco0+Fco20+Fc1i
F_h_initial = Fh20*2 + Fc1i*4 + Fh2o0*2
F_o_initial = Fco0 + Fh2o0 + Fco20*2
F_carbon_paraffin=Fc1[-1]+2*Fc2[-1]+3*Fc3[-1]+4*Fc4[-1]+5*Fc5[-1]+6*Fc6[-1]+7*Fc7[-1]+8*Fc8[-1]+9*Fc9[-1]+10*Fc10[-1]
F_h_paraffin = Fc1[-1]*4 + Fc2[-1]*6 + Fc3[-1]*8 + Fc4[-1]*10 + Fc5[-1]*12 + Fc6[-1]*14 + Fc7[-1]*16 + Fc8[-1]*18 + Fc9[-1]*20 + Fc10[-1]*22
F_carbon_oleffin=Fcc2[-1]*2+Fcc3[-1]*3+Fcc4[-1]*4+Fcc5[-1]*5+Fcc6[-1]*6+Fcc7[-1]*7+Fcc8[-1]*8+Fcc9[-1]*9+Fcc10[-1]*10
F_h_oleffin = Fcc2[-1]*4 + Fcc3[-1]*6 + Fcc4[-1]*8 + Fcc5[-1]*10 + Fcc6[-1]*12 + Fcc7[-1]*14 + Fcc8[-1]*16 + Fcc9[-1]*18 + Fcc10[-1]*20
Hdiff = F_h_initial - F_h_paraffin - F_h_oleffin - 2*Fh2o[-1] - 2*Fh2[-1]
F_carbon_final=Fco2[-1]+F_carbon_paraffin+F_carbon_oleffin+Fco[-1]
Cdiff=F_carbon_initial-F_carbon_final
F_o_final = Fco2[-1]*2 + Fh2o[-1] + Fco[-1]
Odiff = F_o_initial - F_o_final
Mdesiredd = (Fc5*72 + Fc6*86 + Fc7*100 + Fc8*114 + Fc9*128 + Fc10*142 + Fcc5*70 + Fcc6*84 + Fcc7*98 + Fcc8*112 + Fcc9*126 + Fcc10*140)
#(Fc5*72 + Fc6*86 + Fc7*100 + Fc8*114 + Fc9*128 + Fc10*142 + Fcc5*70 + Fcc6*84 + Fcc7*98 + Fcc8*112 + Fcc9*126 + Fcc10*140)
Mass = (Mdesiredd[-1]*Tube_number)/1000
annual_product=((Mass*3600*24*300)/1000)
Total_catalyst = Wx[-1]
print("Annual production = "+str(annual_product)+ " ton gasoline /year")
print("Atom Balance (Ninitial - Nfinal)")
print("Carbon Difference = "+str(Cdiff))
print("Hydrogen Difference = "+str(Hdiff))
print("Oxygen Difference = "+str(Odiff))
print("Tube Number = "+str(Tube_number))
print("Pipe Diameter = "+str(round(D,3))+" m")
print("Required Pipe Length = "+str(round(L,3))+" m")
prop = [str(Tl),str(round(T[-1]-273,2)),str(Pxi),str(round((Ptotal[-1]/100000),2)),str(round(Conversion,3)),
       str(round(Mass_s,3)),str(round(annual_product,1)),str(Tube_number),str(round(D,3)),str(round(L,3)),str(Dp*1000)
       ,str(Tfluid),str(pCat),str(fi),str(round(Tube_number*Wx[-1]/1000,2)),str(Ftotal_initial)]
ddd = pd.DataFrame({"   ":plist})
ddd ["Operating Conditions"] = prop
fig, ax = plt.subplots(figsize=(2.5,6))
ax.axis('tight')
ax.axis('off')
table_data = [ddd.columns] + ddd.values.tolist()
table = ax.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(17)
table.scale(4.5, 5)  # Adjust the table size as needed
#plt.title("Reactor Operating Conditions",fontsize = 14)
plt.show()
