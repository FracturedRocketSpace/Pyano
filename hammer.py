import numpy as np
import config as c
import matplotlib.pyplot as plt

x = np.arange(0, c.length + c.dx, c.dx)
t = np.arange(0, c.tmax + c.dt, c.dt)

g = np.exp(- (x-c.hammerLocation)**2/ (2*c.hammerSize**2))

plt.figure()
plt.plot(x,g)


hammerInteraction = True
hammerDisplacement = np.zeros([len(t)+1])
hammerDisplacement[3] = c.hammerVelocity*c.dt

K=4e8;
p=2.3;

for n in range(3,len(t)):
    if hammerInteraction:
        
        hammerForce = K* abs(hammerDisplacement - dev[c.hammerLocation*len(x),n])**p
        hammerDisplacement[n+1]=2*hammerDisplacement[n]-hammerDisplacement[n-1]-(c.dt**2*hammerForce)/c.hammerMass

        dev[:,n+1] += c.dt**2*len(x)*hammerForce*g/(c.density*c.length)        
        
        if (hammerDisplacement[n]<dev[c.hammerLocation*len(x),n]):
            hammerInteraction = False
    
    
    