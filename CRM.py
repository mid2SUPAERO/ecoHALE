span = 58.7630524 # [m] baseline CRM
W0 = 0.5 * 2.5e6 # [N] (MTOW of B777 is 3e5 kg with fuel)
CT = 9.81 * 17.e-6 # [1/s] (9.81 N/kg * 17e-6 kg/N/s)
R = 14.3e6 # [m] maximum range
M = 0.84 # at cruise

alpha = 3. # [deg.]
rho = 0.38 # [kg/m^3] at 35,000 ft
a = 295.4 # [m/s] at 35,000 ft
v = a * M

CL0 = 0.2
CD0 = 0.015
