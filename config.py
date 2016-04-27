import math


#tension = 670  # String tension [N]
#density = 3.93 /1000/ 0.62;  # Densiuty [kg/m]
#
#c = (tension / density) ** (1 / 2)  # Speed of sound
#eps = 3.82e-5;  # Stiffness
#b1 = 0.5;  # First damping
#b3 = 6.25e-9;  # Third damping
#
#length = 0.62;  # String length [m]
#dx = 0.62 / 50;  # x element size (m)
#
#tmax = 5  # Simulated time [s]
#dt = 1 / (32e3);  # Time step [s]
#
#hammerSize = 0.001  # Length of string hit by hammer
#hammerVelocity = 1  # Initial velocity of string due to hammer
#hammerLocation = 0.2  # Relative distance from the end of the string to hammer striking location
#hammerMass = 0.00297     #Hammer mass in kg
#hammerStiffness = 4.5e9
#hammerExponent = 2.5

# hammerSize = 0.02                   #Length of string hit by hammer
# hammerVelocity = 1                  #Initial velocity of string due to hammer
# hammerLocation = 0.15               #Distance from the end of the string to hammer striking location
# hammerMass = 0.0049                 # hammer mass in kg

format = 1;  # DUnno what this means
CHUNK = 5 * 1024  # CHUNK load size
numChannels = 1;  # = Mono

def selectParameters(note):    
    youngMod = 2e11
    
    if (note==40):
        # C4
        length = 0.657
        d = 1.006e-3
        rho = 7850
        tension = 741
        b1 = 0.5;
        b3 = 6.25e-9

        hammerExponent = 2.8
        hammerLocation = 0.079/length
        hammerMass = 0.00871
        hammerStiffness = 4.84e9
        hammerSize = 0.01
        hammerVelocity = 1
        
        spatialSteps = 50
        tmax = 5
        
        Fs = 32e3
        
        density = rho*math.pi/4*d**2
        vel = (tension/density)**.5
        gyradius = d/4
        eps = gyradius**2 * (youngMod*math.pi/4*d**2/(tension*length**2))
        
        dx = length/spatialSteps
        dt = 1/Fs
        
    elif (note == 42):
        # D4
        length = 0.59
        d = 0.991e-3
        rho = 7850
        tension = 730
        b1 = 0.5;
        b3 = 6.25e-9

        hammerExponent = 2.442
        hammerLocation = 0.071/length
        hammerMass = 0.00858
        hammerStiffness = 7.457e9
        hammerSize = 0.01
        hammerVelocity = 1
        
        spatialSteps = 50
        tmax = 5
        
        Fs = 64e3
        
        density = rho*math.pi/4*d**2
        vel = (tension/density)**.5
        gyradius = d/4
        eps = gyradius**2 * (youngMod*math.pi/4*d**2/(tension*length**2))

        dx = length/spatialSteps
        dt = 1/Fs
    elif (note == 44):
        #E4
        length = 0.529
        d = 0.977e-3
        rho = 7850
        tension = 720
        b1 = 0.5;
        b3 = 6.25e-9

        hammerExponent = 2.468
        hammerLocation = 0.064/length
        hammerMass = 0.00846
        hammerStiffness = 9.523e9
        hammerSize = 0.01
        hammerVelocity = 1
        
        spatialSteps = 50
        tmax = 5
        
        Fs = 64e3
        
        density = rho*math.pi/4*d**2
        vel = (tension/density)**.5
        gyradius = d/4
        eps = gyradius**2 * (youngMod*math.pi/4*d**2/(tension*length**2))
        
        dx = length/spatialSteps
        dt = 1/Fs 
    elif (note==45):
        # F4
        length = 0.501
        d = 0.970e-3
        rho = 7850
        tension = 715
        b1 = 0.5;
        b3 = 6.25e-9

        hammerExponent = 2.482
        hammerLocation = 0.060/length
        hammerMass = 0.00839
        hammerStiffness = 1.076e10
        hammerSize = 0.01
        hammerVelocity = 1
        
        spatialSteps = 50
        tmax = 5
        
        Fs = 64e3
        
        density = rho*math.pi/4*d**2
        vel = (tension/density)**.5
        gyradius = d/4
        eps = gyradius**2 * (youngMod*math.pi/4*d**2/(tension*length**2))
        
        dx = length/spatialSteps
        dt = 1/Fs
    elif (note == 47):
        # G4
        length = 0.450
        d = 0.958e-3
        rho = 7850
        tension = 707
        b1 = 0.5;
        b3 = 6.25e-9

        hammerExponent = 2.512
        hammerLocation = 0.054/length
        hammerMass = 0.00827
        hammerStiffness = 1.374e10
        hammerSize = 0.01
        hammerVelocity = 1
        
        spatialSteps = 50
        tmax = 5
        
        Fs = 64e3
        
        density = rho*math.pi/4*d**2
        vel = (tension/density)**.5
        gyradius = d/4
        eps = gyradius**2 * (youngMod*math.pi/4*d**2/(tension*length**2))
        
        dx = length/spatialSteps
        dt = 1/Fs
    elif (note == 49):
        # A4
        length = 0.404
        d = 0.947e-3
        rho = 7850
        tension = 701
        b1 = 0.5;
        b3 = 6.25e-9

        hammerExponent = 2.543
        hammerLocation = 0.048/length
        hammerMass = 0.00814
        hammerStiffness = 1.755e10
        hammerSize = 0.01
        hammerVelocity = 1
        
        spatialSteps = 50
        tmax = 5
        
        Fs = 64e3
        
        density = rho*math.pi/4*d**2
        vel = (tension/density)**.5
        gyradius = d/4
        eps = gyradius**2 * (youngMod*math.pi/4*d**2/(tension*length**2))
        
        dx = length/spatialSteps
        dt = 1/Fs
    elif (note == 51):
        # B4
        length = 0.363
        d = 0.937e-3
        rho = 7850
        tension = 697
        b1 = 0.5;
        b3 = 6.25e-9

        hammerExponent = 2.576
        hammerLocation = 0.044/length
        hammerMass = 0.00802
        hammerStiffness = 2.241e10
        hammerSize = 0.01
        hammerVelocity = 1
        
        spatialSteps = 40
        tmax = 5
        
        Fs = 64e3
        
        density = rho*math.pi/4*d**2
        vel = (tension/density)**.5
        gyradius = d/4
        eps = gyradius**2 * (youngMod*math.pi/4*d**2/(tension*length**2))
        
        dx = length/spatialSteps
        dt = 1/Fs
    else:
        print ("Error: Unknown note")
        
    return length, tension, b1, b3, hammerExponent, hammerLocation, hammerMass, hammerStiffness, hammerSize, hammerVelocity, dx, tmax, Fs, dt, density, eps, vel
        
        
        