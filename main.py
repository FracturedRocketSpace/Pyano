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
from numba import jit
import locale
locale.setlocale(locale.LC_NUMERIC, 'C')
import sounddevice as sd

x = np.arange(-c.dx, c.length + 2 * c.dx, c.dx)
t = np.arange(0, c.tmax + c.dt, c.dt)

# Initiate arrays for deviation and velocity
dev = np.zeros([len(x), len(t)], dtype=np.float32)

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
hammerDisplacement[10] = c.hammerVelocity*c.dt

A1 = np.zeros((N,N), dtype=np.float32);
A2 = np.zeros((N,N), dtype=np.float32);
A3 = np.zeros((N,N), dtype=np.float32);
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

iter = np.zeros(2, dtype=np.int8);
bridge = int(c.bridgePos * len(x));

# Set samplerate
sd.default.samplerate = c.framerate;
sd.default.latency = 'high'
st = 0
streamer = sd.OutputStream(channels=1, dtype='float32');
CHUNK = streamer.write_available - 1
streamer.start()
streamer2 = sd.OutputStream(channels=1, dtype='float32');
streamer2.start()

# Running the simulation

start = timeit.default_timer()
st=0;
for i in range(3, len(t)):
    if hammerInteraction and hammerDisplacement[i]>0:
        hammerForce = c.hammerStiffness* abs(hammerDisplacement[i] - dev[int(c.hammerLocation*len(x)),i])**c.hammerExponent
        hammerDisplacement[i+1]=2*hammerDisplacement[i]-hammerDisplacement[i-1]-(c.dt**2*hammerForce)/c.hammerMass

        dev[:,i] += c.dt**2*len(x)*hammerForce*g/(c.density*c.length)
        
        if (hammerDisplacement[i]<dev[int(c.hammerLocation*len(x)),i]):
            hammerInteraction = False
            
    dev[:, i] += iterate(dev[:, i - 1],dev[:, i - 2],dev[:, i - 3], A1, A2, A3);

    if((i+1)%CHUNK == 0):
        streamer.write(dev[bridge, st:i]/c.norm)
        streamer2.write(dev[bridge*3, st:i] / c.norm)
        st=i;
        print('Now at ', i + 1, 'of the ', len(t));



print("Program ended in  =", int(timeit.default_timer() - start), "seconds");

sd.wait()
start = timeit.default_timer()
audio=dev[bridge,:]
norm=max(audio);
audio /= norm;
sd.play(audio)
print(norm)
sd.wait()
streamer.close()
print("Program ended in  =", int(timeit.default_timer() - start), "seconds");

print("Calculating and plotting spectrum", flush=True)
spectrum = scipy.fftpack.fft(audio)
#spectrum = scipy.fftpack.fft(np.sin(2*np.pi*1000*t))
freq= np.linspace(0,1/(2*c.dt),len(t)/2)
plt.figure()
plt.plot(freq, np.abs(spectrum[:len(t)/2]))

plt.xlim(20,10000)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Intensity (a.u.)")

# plot/animate results: string animation, frequency spectrum
X, T = np.meshgrid(x, t)

plt.figure()
plt.plot(t, audio * 127)
plt.show()