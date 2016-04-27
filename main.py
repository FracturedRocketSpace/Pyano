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
from config import selectParameters

def simulate(note):
    #note = input("input note")
    length, tension, b1, b3, hammerExponent, hammerLocation, hammerMass, hammerStiffness, hammerSize, hammerVelocity, dx, tmax, Fs, dt, density, eps, vel = selectParameters(int(note))  
    
    x = np.arange(-dx, length + 2 * dx, dx)
    t = np.arange(0, tmax + dt, dt)
    
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
    D = 1 + b1 * dt + 2 * b3 / dt;
    r = vel * dt / dx;
    N = len(x)
    
    a1 = (2 - 2 * r ** 2 + b3 / dt - 6 * eps * (N ** 2) * (r ** 2)) / D;
    a2 = (-1 + b1 * dt + 2 * b3 / dt) / D;
    a3 = (r ** 2 * (1 + 4 * eps * (N ** 2))) / D;
    a4 = (b3 / dt - eps * (N ** 2) * (r ** 2)) / D;
    a5 = (- b3 / dt) / D;
    
    # Define spatial extent of hammer
    g = np.exp(- (x-hammerLocation*length)**2/ (2*hammerSize**2))
    g[0]=0; g[-1]=0;
    # Initiate hammer variables
    hammerInteraction = True
    hammerDisplacement = np.zeros([len(t)+1])
    hammerDisplacement[3] = hammerVelocity*dt
    dev[:, 0] += g* hammerVelocity * dt
    dev[:, 1] += g* hammerVelocity * dt
    dev[:, 2] += g* hammerVelocity * dt
    
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
            hammerForce = hammerStiffness* abs(hammerDisplacement[i] - dev[int(hammerLocation*len(x)),i])**hammerExponent
            hammerDisplacement[i+1]=2*hammerDisplacement[i]-hammerDisplacement[i-1]-(dt**2*hammerForce)/hammerMass
    
            dev[:,i] += dt**2*len(x)*hammerForce*g/(density*length)
            
            if (hammerDisplacement[i]<dev[int(hammerLocation*len(x)),i]):
                hammerInteraction = False
                
        dev[:, i] = iterate(dev[:, i - 1],dev[:, i - 2],dev[:, i - 3], A1, A2, A3);
        if(i%1000 == 0):
            print('Now at ', i + 1, 'of the ', len(t));
        
    print("Program ended in  =", int(timeit.default_timer() - start), "seconds");
    
    # Get sound output
    bridgePos=.5
    audio = dev[int(bridgePos * len(x) ), :];
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
                    rate=int(Fs),
                    output=True)
    
    # output sounds
    start = timeit.default_timer()
    stream.write(audio_out)
    print("Program ended in  =", int(timeit.default_timer() - start), "seconds", flush=True);
    
    # Stop the audio output
    stream.stop_stream()
    stream.close()
    p.terminate();

def plotSpectrum(audio, dt, t):
    print("Calculating and plotting spectrum", flush=True)
    spectrum = scipy.fftpack.fft(audio)
    #spectrum = scipy.fftpack.fft(np.sin(2*np.pi*1000*t))
    freq= np.linspace(0,1/(2*dt),len(t)/2)
    plt.figure()
    plt.plot(freq, np.abs(spectrum[:len(t)/2]))
    
    plt.xlim(20,10000)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Intensity (a.u.)")
    

# plot/animate results: string animation, frequency spectrum
#X, T = np.meshgrid(x, t)

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

#plt.figure()
#plt.plot(t, audio * 127)
#plt.show()

#UI for playing music
whiteKeys = np.array(['a','s','d','f','g','h','j']);
blackKeys = np.array(['w','e','t','y','u']);

whiteNotes = np.array([40,42,44,45,47,49,51]);

numWhiteKeys = len(whiteNotes);
whiteKeyWidth = 10;
blackKeyWidth = int(whiteKeyWidth/2);
whiteKeyHeight = 25;
blackKeyHeight = int(whiteKeyHeight/2);

#experimentally determined values for char to pixel size conversion
charWidth = 7;
charHeight = 15;

keyBorder = 5;
padding = 2;

def testFunction(n):
    #simulating and playing a sound goes here
    print(n);

def onPressed(w):
    w.invoke();
    w.configure(relief='sunken');

def onReleased(w):
    w.configure(relief='raised')

root = tk.Tk();
root.geometry(str(int(numWhiteKeys*(whiteKeyWidth*charWidth+2*(padding+keyBorder+1)))) + "x" + str(int(whiteKeyHeight*charHeight+2*(padding+keyBorder+1))));
root.configure(background='#B22222');

#white keys
for n in range(numWhiteKeys):
    w = tk.Button(root, borderwidth = keyBorder, background='white', height = whiteKeyHeight, width=whiteKeyWidth, command = (lambda n=n: simulate(whiteNotes[n])));

    #bind a keypress and release to the button
    root.bind(whiteKeys[n], (lambda event, w=w: onPressed(w)));
    root.bind("<KeyRelease-" + whiteKeys[n] + ">", (lambda event, w=w: onReleased(w)));

    #place button within the window
    w.pack(side='left');
    
#black keys for middle C upwards
currentKey = 0;
for n in range(numWhiteKeys-1):
    #if there is a black note between the white notes
    if(whiteNotes[n+1] - whiteNotes[n] != 1):
        w = tk.Button(root, borderwidth = keyBorder, background='black', height = blackKeyHeight, width=blackKeyWidth, command = (lambda n=n: simulate(whiteNotes[n]+1)));
    
        #bind a keypress and release to the button
        root.bind(blackKeys[currentKey], (lambda event, w=w: onPressed(w)));
        root.bind("<KeyRelease-" + blackKeys[currentKey] + ">", (lambda event, w=w: onReleased(w))); 
        currentKey += 1;
        
        w.place(relx=(n+1)/numWhiteKeys, rely=0.5, anchor='s');

root.mainloop();