import numpy as np
import tkinter as tk
import config as c
import scipy.sparse
import scipy.fftpack
import timeit
from numba import jit
from config import selectParameters
import locale
locale.setlocale(locale.LC_NUMERIC, 'C')
import sounddevice as sd
from multiprocessing import Pool

@jit( nopython=True )
def iterate(dev1,dev2,dev3,A1,A2):
    dev = np.dot(A1,dev1) + np.dot(A2,dev2);
    # 2nd
    dev[0] = -dev[2];
    dev[-1] = -dev[-3];    
    return dev

def simulate(note):
    start = timeit.default_timer()
    #note = input("input note")
    length, tension, b1, b2, kap, hammerExponent, hammerLocation, hammerMass, hammerStiffness, hammerSize, hammerVelocity, dx, tmax, Fs, dt, density, eps, vel = selectParameters(int(note))  
    
    #set small x for speed purposes    
    dx = length/25
    
    x = np.arange(-dx, length + 2 * dx, dx)
    t = np.arange(0, tmax + dt, dt)
    
    # Initiate arrays for deviation and velocity
    dev = np.zeros([len(x), len(t)], dtype='float32')
    
    # Create matrices
    la = vel*dt/dx;
    mu = kap*dt/(dx**2);
    D = (1+b1*dt)
    N = len(x)
    
    a10 = (2 - 2 * la**2 - 6 * mu**2 - 4 * b2 * mu / kap) / D;
    a11 = (la**2 + 4 * mu**2 + 2 * b2 * mu / kap) / D;
    a12 = (-mu**2) / D;
    a20 = (-1 + 4 * b2 * mu / kap + b1 * dt) / D;
    a21 = (-2 * b2 * mu / kap) / D;
    
    # Define spatial extent of hammer
    #GAUSS
    g = np.exp(- (x-hammerLocation*length)**2/ (2*hammerSize**2))    
    g[0]=0; g[-1]=0;
    
    # Initiate hammer variables
    hammerInteraction = True
    hammerDisplacement = np.zeros([len(t)+1])
    hammerDisplacement[3] = hammerVelocity*dt
    
    #compute matrices
    A1 = np.zeros((N,N),dtype = 'float32');
    A2 = np.zeros((N,N),dtype = 'float32');
    i,j = np.indices(A1.shape);
    
    A1[i==j] = a10;
    A1[i==j-1] = A1[i==j+1] = a11;
    A1[i==j-2] = A1[i==j+2] = a12;
    
    A2[i==j] = a20;
    A2[i==j-1] = A2[i==j+1] = a21;
    
    #keep edges at zero
    A1[0,:] = A1[-1,:] = 0;
    A1[1,:] = A1[-2,:] = 0;
    
    A2[0,:] = A2[-1,:] = 0;
    A2[1,:] = A2[-2,:] = 0;
    #
    streamer = sd.OutputStream(samplerate=Fs, channels=1, dtype='float32');
    CHUNK = max([streamer.write_available - 1,c.minCHUNK])
    streamer.start();
    print("Initialized note", note, "in", timeit.default_timer() - start, "seconds", flush=True);

    # Running the simulation
    start = timeit.default_timer()
    st = 0;
    for i in range(3, len(t)):
        if hammerInteraction:       
            hammerForce = hammerStiffness* abs(hammerDisplacement[i] - dev[int(hammerLocation*len(x)),i])**hammerExponent
            hammerDisplacement[i+1]=2*hammerDisplacement[i]-hammerDisplacement[i-1]-(dt**2*hammerForce)/hammerMass
    
            dev[:,i] += dt**2*len(x)*hammerForce*g/(density*length)
            # Hammer could move it
            dev[:,1] = dev[:,-2] = 0;
            
            if (hammerDisplacement[i]<dev[int(hammerLocation*len(x)),i]):
                hammerInteraction = False
                
        dev[:, i] += iterate(dev[:, i - 1],dev[:, i - 2],dev[:, i - 3], A1, A2);

        if ((i + 1) % CHUNK == 0):
            if(st==0):
                norm = max(abs(dev[c.bridgePos, st:i]));
            streamer.write(dev[c.bridgePos, st:i] / norm);
            st = i;

    print("Simulated ",tmax, "seconds of note", note, "in", timeit.default_timer() - start, "seconds", flush=True);
    
    if c.spectrum:
        audio = dev[c.bridgePos, :]
        plotSpectrum(audio, dt, t)

# Functions
def plotSpectrum(audio, dt, t):
    import matplotlib.pyplot as plt
    print("Calculating and plotting spectrum", flush=True)
    spectrum = scipy.fftpack.fft(audio)
    freq= np.linspace(0,1/(2*dt),len(t)/2)
    plt.figure()
    plt.plot(freq, np.abs(spectrum[:len(t)/2]))
    
    plt.xlim(20,10000)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Intensity (a.u.)")
    plt.show()
    
def buttonPressed(n):
    print(n);
    pool.apply_async(simulate,(n,));

def onPressed(w):
    w.invoke();
    w.configure(relief='sunken');

def onReleased(w):
    w.configure(relief='raised')    

if __name__ == '__main__':
    with Pool(processes=c.numProcesses) as pool:

        
        
        #UI for playing music
        whiteKeys = np.array(['a','s','d','f','g','h','j','k']);
        blackKeys = np.array(['w','e','t','y','u']);
        
        whiteNotes = np.array([40,42,44,45,47,49,51,52]);
        
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
            w = tk.Button(root, borderwidth = keyBorder, background='white', height = whiteKeyHeight, width=whiteKeyWidth, command = (lambda n=n: buttonPressed(whiteNotes[n])));
        
            #bind a keypress and release to the button
            root.bind(whiteKeys[n], (lambda event, w=w: onPressed(w)));
            root.bind("<KeyRelease-" + whiteKeys[n] + ">", (lambda event, w=w: onReleased(w)));
        
            #place button within the window
            w.place(relx=(n+1)/numWhiteKeys, rely=0.0, anchor='ne');
            
        #black keys for middle C upwards
        currentKey = 0;
        for n in range(numWhiteKeys-1):
            #if there is a black note between the white notes
            if(whiteNotes[n+1] - whiteNotes[n] != 1):
                w = tk.Button(root, borderwidth = keyBorder, background='black', height = blackKeyHeight, width=blackKeyWidth, command = (lambda n=n: buttonPressed(whiteNotes[n]+1)));
            
                #bind a keypress and release to the button
                root.bind(blackKeys[currentKey], (lambda event, w=w: onPressed(w)));
                root.bind("<KeyRelease-" + blackKeys[currentKey] + ">", (lambda event, w=w: onReleased(w))); 
                currentKey += 1;
                
                w.place(relx=(n+1)/numWhiteKeys, rely=0.5, anchor='s');
        
        root.mainloop();