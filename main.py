import numpy as np
import math
import tkinter as tk
import config as c
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for projection='3d'!
from matplotlib import cm
from matplotlib import animation


#theoretical values
speed = np.sqrt(c.tension/c.density)
f0 = speed/(2*c.length);
inharmonicity = math.pi**2*c.elasticModulus*c.crossArea*c.gyration**2/(c.tension*c.length**2);

#inharmonic partial frequencies
order = 1;
fn = order*f0*np.sqrt(1+inharmonicity*order**2);

x = np.arange(0, c.length + c.dx, c.dx)
t = np.arange(0, c.tmax + c.dt, c.dt)

#Initiate arrays for deviation and velcity
dev = np.zeros([len(x),len(t)])
vel = np.zeros([len(x),len(t)])

# Initial conditino: Strike string with hammer
vel[int(c.hammerLocation/c.length*len(x) ): int((c.hammerLocation+c.hammerSize)/c.length*len(x)), 0] = c.hammerVelocity;

def calcForce(dev):
    devPadded = np.zeros([len(x)+4])
    devPadded[2:-2]=dev

    force1 = c.tension/c.dx**2* (devPadded[1:-3] - 2*devPadded[2:-2] + devPadded[3:-1] ) 
    force2 = c.elasticModulus*c.crossArea*c.gyration**2/c.dx**4 * (devPadded[0:-4] - 4*devPadded[1:-3] + 6* devPadded[2:-2] - 4*devPadded[3:-1] + devPadded[4::])   
    force = force1+force2  
    print( np.isnan(force1).any() )
    print( np.isnan(force2).any() )
    return force, force1, force2

def iterate(dev, vel, force):
    
    accelleration = force/ (c.density*c.dx)

    devNew = dev + vel*c.dt + 1/2*accelleration*c.dt**2;
    velNew = vel + 1/2*accelleration*c.dt
    
    force, force1, force2 = calcForce(devNew)
    
    velNew += 1/2 * force/(c.density*c.dx) * c.dt
        
    return devNew, velNew, force, force1, force2
    
# force for first iteration
force, force1, force2 = calcForce(dev[:,0])
for i in range(1, len(t)):
    dev[:,i], vel[:,i], force, force1, force2 = iterate(dev[:,i-1], vel[:,i-1], force)

#plot/animate results: string animation, frequency spectrum
X, T =np.meshgrid(x,t)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X,T ,dev.T, rstride=10, cstride=10,cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.view_init(90, 90); # Top view
plt.xlabel("Position")
plt.ylabel("Time")
plt.title("Implicit method")
fig.colorbar(surf, shrink=0.5, aspect=5)
# Hide z-axis
ax.w_zaxis.line.set_lw(0.)
ax.set_zticks([])
    

#output sounds

#UI for playing music
numKeys = 9;
keyWidth = 10;
keyHeight = 200;
keyBorder = 5;
padding = 2;

keys = np.array(['a','s','d','f','g','h','j','k','l'])

def testFunction(n):
    #simulating and playing a sound goes here
    print(n);

def onPressed(w):
    w.invoke(); 
    w.configure(relief='sunken');
    
def onReleased(w):
    w.configure(relief='raised')
    
root = tk.Tk();
root.geometry(str(numKeys*(keyWidth*8+keyBorder+1)) + "x" + str(keyHeight));
root.configure(background='#B22222');

for n in range(numKeys):
    w = tk.Button(root, borderwidth = keyBorder, background='white', height = keyHeight, width=keyWidth, command = (lambda n=n: testFunction(n)));
    
    #bind a keypress and release to the button
    root.bind(keys[n], (lambda event, w=w: onPressed(w)));
    root.bind("<KeyRelease-" + keys[n] + ">", (lambda event, w=w: onReleased(w)));
    
    #place button within the window
    w.pack(side='left');

root.mainloop();