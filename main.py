import numpy as np
import math
import tkinter as tk
import config as c
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for projection='3d'!
from matplotlib import cm
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math
import scipy.sparse
import scipy.fftpack
import timeit
import pyaudio
from numba import jit

x = np.arange(-c.dx, c.length + 2 * c.dx, c.dx)
t = np.arange(0, c.tmax + c.dt, c.dt)

# Initiate arrays for deviation and velocity
dev = np.zeros([len(x), len(t)])

## Initial conditino: Strike string with hammer
## Gaussian hammer
#mean = c.hammerLocation;
#variance = c.hammerSize;
#sigma = math.sqrt(variance);
#vel[:, 0] = c.hammerVelocity * mlab.normpdf(x, mean, sigma);
## Pulse hammer
## vel[int(c.hammerLocation/c.length*len(x) ): int((c.hammerLocation+c.hammerSize)/c.length*len(x)), 0] = c.hammerVelocity;
#
## force for first iteration
#dev[:, 0] += vel[:, 0] * c.dt
#dev[:, 1] += vel[:, 0] * c.dt
#dev[:, 2] += vel[:, 0] * c.dt

# Create matrices
D = 1 + c.b1 * c.dt + 2 * c.b3 / c.dt;
r = c.c * c.dt / c.dx;
N = len(x)

a1 = (2 - 2 * r ** 2 + c.b3 / c.dt - 6 * c.eps * (N ** 2) * (r ** 2)) / D;
a2 = (-1 + c.b1 * c.dt + 2 * c.b3 / c.dt) / D;
a3 = (r ** 2 * (1 + 4 * c.eps * (N ** 2))) / D;
a4 = (c.b3 / c.dt - c.eps * (N ** 2) * (r ** 2)) / D;
a5 = (- c.b3 / c.dt) / D;

# Define spatial extent of hammer
g = np.exp(- (x-c.hammerLocation*c.length)**2/ (2*c.hammerSize**2))
# Initiate hammer variables
hammerInteraction = True
hammerDisplacement = np.zeros([len(t)+1])
hammerDisplacement[3] = c.hammerVelocity*c.dt
dev[:, 0] += g* c.hammerVelocity * c.dt
dev[:, 1] += g* c.hammerVelocity * c.dt
dev[:, 2] += g* c.hammerVelocity * c.dt

A1 = np.zeros((N,N));
A2 = np.zeros((N,N));
A3 = np.zeros((N,N));
i,j = np.indices(A1.shape);

A1[i==j] = a1;
A1[i==j-1] = A1[i==j+1] = a3;
A1[i==j-2] = A1[i==j+2] = a4;
A1[0,:] = A1[-1,:] = 0;

A2[i==j] = a2;
A2[i==j-1] = A2[i==j+1] = a5;
A2[0,:] = A2[-1,:] = 0;

A3[i==j] = a5;
A3[0,:] = A3[-1,:] = 0;

@jit( nopython=True )
def iterate(dev1,dev2,dev3,A1,A2,A3):
    dev = np.dot(A1,dev1) + np.dot(A2,dev2) + np.dot(A3,dev3);
    # end zero
    dev[1] = 0;
    dev[-2] = 0;
    # 2nd
    dev[0] = -dev[2];
    dev[-1] = -dev[-3];    
    return dev

# Running the simulation
start = timeit.default_timer()
for i in range(3, len(t)):
    if hammerInteraction:       
        hammerForce = c.hammerStiffness* abs(hammerDisplacement[i] - dev[int(c.hammerLocation*len(x)),i])**c.hammerExponent
        hammerDisplacement[i+1]=2*hammerDisplacement[i]-hammerDisplacement[i-1]-(c.dt**2*hammerForce)/c.hammerMass

        dev[:,i] += c.dt**2*len(x)*hammerForce*g/(c.density*c.length)
        
        if (hammerDisplacement[i]<dev[int(c.hammerLocation*len(x)),i]):
            hammerInteraction = False
            
    dev[:, i] = iterate(dev[:, i - 1],dev[:, i - 2],dev[:, i - 3], A1, A2, A3);
    if(i%1000 == 0):
        print('Now at ', i + 1, 'of the ', len(t));
    
print("Program ended in  =", int(timeit.default_timer() - start), "seconds");

# Get sound output
audio = dev[int(c.bridgePos * len(x) ), :];
print(len(audio))
# Normalize and convert
norm = max(abs(audio));
audio = audio / norm;
audio_out = np.array(audio * 127 + 128, dtype=np.int8).view('c');
# Init sound
p = pyaudio.PyAudio()
# Open stream to audio device
# Format: Array type. Int32 or float32 for example. 1 = float32?
# Channels. Number of channels. 1=mono, 2=stereo
# Rate: The sampling rate
# Output: True of course as we want output
stream = p.open(format=p.get_format_from_width(c.format),
                channels=c.numChannels,
                rate=c.framerate,
                output=True)

# output sounds
start = timeit.default_timer()
stream.write(audio_out)
print("Program ended in  =", int(timeit.default_timer() - start), "seconds");

# Stop the audio output
stream.stop_stream()
stream.close()

spectrum = scipy.fftpack.fft(audio)
#spectrum = scipy.fftpack.fft(np.sin(2*np.pi*1000*t))
freq= np.linspace(0,1/(2*c.dt),len(t)/2)
plt.figure()
plt.plot(freq, np.abs(spectrum[:len(t)/2]))

plt.xlim(20,10000)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Intensity (a.u.)")
p.terminate();


#UI for playing music
# numKeys = 9;
# keyWidth = 10;
# keyHeight = 200;
# keyBorder = 5;
# padding = 2;
#
# keys = np.array(['a','s','d','f','g','h','j','k','l'])
#
# def testFunction(n):
#     #simulating and playing a sound goes here
#     print(n);
#
# def onPressed(w):
#     w.invoke();
#     w.configure(relief='sunken');
#
# def onReleased(w):
#     w.configure(relief='raised')
#
# root = tk.Tk();
# root.geometry(str(numKeys*(keyWidth*8+keyBorder+1)) + "x" + str(keyHeight));
# root.configure(background='#B22222');
#
# for n in range(numKeys):
#     w = tk.Button(root, borderwidth = keyBorder, background='white', height = keyHeight, width=keyWidth, command = (lambda n=n: testFunction(n)));
#
#     #bind a keypress and release to the button
#     root.bind(keys[n], (lambda event, w=w: onPressed(w)));
#     root.bind("<KeyRelease-" + keys[n] + ">", (lambda event, w=w: onReleased(w)));
#
#     #place button within the window
#     w.pack(side='left');
#
# root.mainloop();

# plot/animate results: string animation, frequency spectrum
X, T = np.meshgrid(x, t)

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(X,T ,dev.T, rstride=10, cstride=10,cmap=cm.coolwarm, linewidth=0, antialiased=False)
# ax.view_init(90, 90); # Top view
# plt.xlabel("Position")
# plt.ylabel("Time")
# plt.title("Implicit method")
# fig.colorbar(surf, shrink=0.5, aspect=5)
# Hide z-axis
# ax.w_zaxis.line.set_lw(0.)
# ax.set_zticks([])

plt.figure()
plt.plot(t, audio * 127)
plt.show()