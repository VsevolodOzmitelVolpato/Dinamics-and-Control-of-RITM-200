import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sc

#dichiaro i reactivity coefficients [pcm/C]
af=-2e-5
am=-10e-5 #pcm/°C 
ah=-0.007 #pcm/cm
   
#dichiaro stati stazionari
Tf0=550
Tm0=310
h0=0
Tf,Tm,h=input("Insersci i valori di Tf,Tm,h: ").split(" ")

#calcolo la reactivity
rho= lambda TF,TM, H: af*(int(TF)-Tf0)+am*(int(TM)-Tm0)+ah*(float(H)-h0) 
print(rho(Tf,Tm,h))
BETA = 0.0065  #frazione delayed neutrons
PROMPT_LIFETIME = 1e-4 #mean lifetime 
LAMBDAS = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01])
BETAS_I = BETA * np.array([0.033, 0.219, 0.196, 0.395, 0.115, 0.042])

#imposto le equazioni differenziali
def reactor_dinamics(t,y):
    n=y[0] # y è il vettore delle incognite
    c=y[1:]
    dndt=((rho(Tf,Tm,h) - BETA) / PROMPT_LIFETIME) * n + np.dot(LAMBDAS, c) #bilancio densità neutronica
    dcidt = (BETAS_I / PROMPT_LIFETIME) * n - LAMBDAS * c #bilancio concentrazione di precursori
    return [dndt,*dcidt]
    
#imposto le condizioni a contorno (dallo stato stazionario)
n0 = 1.0
c0 = (BETAS_I / (LAMBDAS * PROMPT_LIFETIME)) * n0
y0 = [n0, *c0]

#risolvo il sistema
sol=sc.solve_ivp(reactor_dinamics, [1,10], y0, method='LSODA')

#stampo la soluzione di n
print("Solution computed successfully!")
print(f"Time points: {len(sol.t)}")
print(f"Neutron density at t=0: {sol.y[0, 0]:.6f}")
print(f"Neutron density at t=10: {sol.y[0, -1]:.6f}")

#plotto n, scala logaritmica
plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], label='Neutron density (n)', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Neutron density')
plt.title('Reactor Dynamics Simulation')
plt.grid(True, alpha=0.3)
plt.legend()

plt.show()
