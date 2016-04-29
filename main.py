import numpy as np
import tkinter as tk
import scipy.sparse
import scipy.fftpack
import timeit
from numba import jit
import locale
locale.setlocale(locale.LC_NUMERIC, 'C')
import sounddevice as sd
from multiprocessing import Pool
import matplotlib.pyplot as plt

import config as c
from config import selectParameters

@jit( nopython = True, cache = True )
def iterate(dev1,dev2,dev3,A1,A2,A3):
    dev = np.dot(A1,dev1) + np.dot(A2,dev2) + np.dot(A3,dev3);

    # 2nd boundary condition
    dev[0] = -dev[2];
    dev[-1] = -dev[-3];

    return dev

def simulate(note):
    # Start timer for initialisation
    start = timeit.default_timer()
    
    # Get parameters from config file
    length, tension, b1, b3, hammerExponent, hammerLocation, hammerMass, hammerStiffness, hammerSize, hammerVelocity, dx, tmax, Fs, dt, density, eps, vel = selectParameters(int(note))

    x = np.arange(-dx, length + 2 * dx, dx)
    t = np.arange(0, tmax + dt, dt)

    # Save deviation in this list to plot spectrum
    devSave=[];

    # Define spatial extent of hammer
    #GAUSS
    g = np.exp( - (x - hammerLocation * length)**2 / (2 * hammerSize**2))
    g[0] = 0; g[-1] = 0;

    # Initiate hammer variables
    hammerInteraction = True
    hammerDisplacement = np.zeros([len(t) + 1])
    hammerDisplacement[3] = hammerVelocity * dt

    # Compute matrix entry values
    D = 1 + b1 * dt + 2 * b3 / dt;
    r = vel * dt / dx;
    N = len(x)
    
    a1 = (2 - 2 * r ** 2 + b3 / dt - 6 * eps * (N ** 2) * (r ** 2)) / D;
    a2 = (-1 + b1 * dt + 2 * b3 / dt) / D;
    a3 = (r ** 2 * (1 + 4 * eps * (N ** 2))) / D;
    a4 = (b3 / dt - eps * (N ** 2) * (r ** 2)) / D;
    a5 = (- b3 / dt) / D;
    
    # Construct matrices
    A1 = np.zeros((N,N), dtype='float32');
    A2 = np.zeros((N,N), dtype='float32');
    A3 = np.zeros((N,N), dtype='float32');
    i,j = np.indices(A1.shape);
    
    A1[i==j] = a1;
    A1[i==j-1] = A1[i==j+1] = a3;
    A1[i==j-2] = A1[i==j+2] = a4;
    A1[0,:] = A1[-1,:] = 0;
    A1[1,:] = A1[-2,:] = 0;
    
    A2[i==j] = a2;
    A2[i==j-1] = A2[i==j+1] = a5;
    A2[0,:] = A2[-1,:] = 0;
    A2[1,:] = A2[-2,:] = 0;
    
    A3[i==j] = a5;
    A3[0,:] = A3[-1,:] = 0;
    A3[1,:] = A3[-2,:] = 0;
    
    # Start the streamer
    streamer = sd.OutputStream(samplerate=Fs, channels=1, dtype='float32');
    streamer.start();
    CHUNK = max([streamer.write_available - 1,c.minCHUNK])
    # Initialize deviation vector
    dev = np.zeros([len(x), CHUNK], dtype='float32')
    iter = 3;
    norm = 0.0;
    # Done!
    print("Initialized note", note, "in", timeit.default_timer() - start, "seconds", flush=True);
    
    # Running the simulation
    start = timeit.default_timer()
    for i in range(3, len(t)):
        if hammerInteraction:       
            hammerForce = hammerStiffness * abs( hammerDisplacement[i] - dev[int(hammerLocation*  len(x)),i] )**hammerExponent
            hammerDisplacement[i+1] = 2 * hammerDisplacement[i] - hammerDisplacement[i-1] - (dt**2 * hammerForce) / hammerMass
            # Wire deviation due to hammer interaction
            dev[:,iter] = dt**2 * len(x) * hammerForce * g / (density * length)
            # Hammer could move it
            dev[1, iter] = dev[-2, iter] = 0;
            # Stop hammering
            if ( hammerDisplacement[i] < dev[int(hammerLocation * len(x)),i] ):
                hammerInteraction = False
            # Wire deviation due to other terms
            dev[:, iter] += iterate(dev[:, iter - 1],dev[:, iter - 2],dev[:, iter - 3], A1, A2, A3);
        else:
            dev[:, iter] = iterate(dev[:, iter - 1],dev[:, iter - 2],dev[:, iter - 3], A1, A2, A3);

        iter+=1;
        # Play sound if enough sound samples are generated
        if (iter % CHUNK == 0):
            # Compute normalasation factor
            if(norm == 0.0):
                norm = max(abs(dev[c.bridgePos])) / c.synthMode
            elif(t[i] > c.dampTime):
                norm *= (1 + c.damping * CHUNK)
            # Play sound and reset iter
            streamer.write(dev[c.bridgePos] / norm);
            iter = 0;
            # Save sound samples if spectrum is required
            if c.spectrum:
                devSave.append(dev[c.bridgePos,:] / norm)
    print("Simulated ",tmax, "seconds of note", note, "in", timeit.default_timer() - start, "seconds", flush=True);
    
    # Construct vector containing all sound samples and plot spectrum
    if c.spectrum:
        audio = np.hstack(devSave)
        plotSpectrum(audio, dt, t, note)

# Functions
def plotSpectrum(audio, dt, t, note):
    print("Calculating and plotting spectrum", flush=True)
    spectrum = scipy.fftpack.fft(audio)
    absSpectrum = np.abs(spectrum);
    freq= np.linspace(0, 1/(2*dt), len(t)/2)  
    
    print("Fundamental frequency peak at %.f Hz" % freq[np.min( np.argmax( absSpectrum[ 0 : 1.5 * np.min(np.argwhere(absSpectrum > 3000)) ] ) )] , flush=True);

    plt.plot(freq, absSpectrum[:len(t)/2])
    plt.xlim(20,2500)
    plt.title(note)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Intensity (a.u.)")
    plt.show()

# Create keyboard with keys to run simulation
if __name__ == '__main__':
    def buttonPressed(n):
        print("Note: " , n);
        pool.apply_async(simulate,(n,));

    def onPressed(w):
        w.invoke();
        w.configure(relief='sunken');

    def onReleased(w):
        w.configure(relief='raised')

    with Pool(processes=c.numProcesses, maxtasksperchild=c.numTasks) as pool:
        #UI for playing music
        whiteKeys = np.array(['A','S','D','F','G','H','J','a','s','d','f','g','h','j','k']);
        blackKeys = np.array(['W','E', 'T','Y','U','w','e','t','y','u']);
        
        whiteNotes = np.array([28,30,32,33,35,37,39,40,42,44,45,47,49,51,52]);
        
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
        
        root = tk.Tk();
        root.geometry(str(int(numWhiteKeys*(whiteKeyWidth*charWidth+2*(padding+keyBorder+1)))) + "x" + str(int(whiteKeyHeight*charHeight+2*(padding+keyBorder+1))));
        root.configure(background='#B22222');
        
        #white keys
        for n in range(numWhiteKeys):
            w = tk.Button(root, borderwidth = keyBorder, text='\n\n\n\n\n\n\n\n\n\n\n\n\n\n'+whiteKeys[n], background='white', height = whiteKeyHeight, width=whiteKeyWidth, command = (lambda n=n: buttonPressed(whiteNotes[n])));
        
            #bind a keypress and release to the button
            root.bind(whiteKeys[n], (lambda event, w=w: onPressed(w)));
            root.bind("<KeyRelease-" + whiteKeys[n] + ">", (lambda event, w=w: onReleased(w)));
        
            #place button within the window
            w.place(relx=(n+1)/numWhiteKeys, rely=0.0, anchor='ne');
            
        #black keys
        currentKey = 0;
        for n in range(numWhiteKeys-1):
            #if there is a black note between the white notes
            if(whiteNotes[n+1] - whiteNotes[n] != 1):
                w = tk.Button(root, borderwidth = keyBorder, text=blackKeys[currentKey], foreground='white', background='black', height = blackKeyHeight, width=blackKeyWidth, command = (lambda n=n: buttonPressed(whiteNotes[n]+1)));
            
                #bind a keypress and release to the button
                root.bind(blackKeys[currentKey], (lambda event, w=w: onPressed(w)));
                root.bind("<KeyRelease-" + blackKeys[currentKey] + ">", (lambda event, w=w: onReleased(w))); 
                currentKey += 1;
                
                w.place(relx=(n+1)/numWhiteKeys, rely=0.5, anchor='s');
        
        root.mainloop();