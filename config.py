import math

format = 1;  # DUnno what this means
CHUNK = 5 * 1024  # CHUNK load size
numChannels = 1;  # = Mono

norm = 5e-4

def selectParameters(note):    
    youngMod = 2e11
    rho = 7850
    b1 = 0.5;
    b3 = 1e-9 #6.25e-9
    tmax = 1
    Fs = 64e3
    hammerSize = 0.01
    hammerVelocity = 1
    
    if (note==40):
        # C4
        length = 0.657
        d = 1.006e-3
        tension = 741
            
        hammerExponent = 2.418 #2.8
        hammerLocation = 0.079/length
        hammerMass = 0.00871
        hammerStiffness = 5.840e9 #4.84e9
        
        spatialSteps = 50
        dx = length/spatialSteps
        dt = 1/Fs
        
        density = rho*math.pi/4*d**2
        vel = (tension/density)**.5
        gyradius = d/4
        eps = gyradius**2 * (youngMod*math.pi/4*d**2/(tension*length**2))
        
    elif (note == 41):
        # Cd4
        length = 0.622
        d = 0.999e-3
        tension = 735

        hammerExponent = 2.430
        hammerLocation = 0.075/length
        hammerMass = 0.00864
        hammerStiffness = 6.599e9
        
        spatialSteps = 50
        dx = length/spatialSteps
        dt = 1/Fs
        
        density = rho*math.pi/4*d**2
        vel = (tension/density)**.5
        gyradius = d/4
        eps = gyradius**2 * (youngMod*math.pi/4*d**2/(tension*length**2))
        
    elif (note == 42):
        # D4
        length = 0.59
        d = 0.991e-3
        tension = 730

        hammerExponent = 2.442
        hammerLocation = 0.071/length
        hammerMass = 0.00858
        hammerStiffness = 7.457e9
        
        spatialSteps = 50
        dx = length/spatialSteps
        dt = 1/Fs
        
        density = rho*math.pi/4*d**2
        vel = (tension/density)**.5
        gyradius = d/4
        eps = gyradius**2 * (youngMod*math.pi/4*d**2/(tension*length**2))
        
    elif (note == 43):
        #Dd4   
        length = 0.559
        d = 0.984e-3
        tension = 725

        hammerExponent = 2.455
        hammerLocation = 0.067/length
        hammerMass = 0.00852
        hammerStiffness = 8.427e9
        
        spatialSteps = 50
        dx = length/spatialSteps
        dt = 1/Fs
        
        density = rho*math.pi/4*d**2
        vel = (tension/density)**.5
        gyradius = d/4
        eps = gyradius**2 * (youngMod*math.pi/4*d**2/(tension*length**2))

    elif (note == 44):
        #E4
        length = 0.529
        d = 0.977e-3
        tension = 720

        hammerExponent = 2.468
        hammerLocation = 0.064/length
        hammerMass = 0.00846
        hammerStiffness = 9.523e9
        
        spatialSteps = 50
        dx = length/spatialSteps
        dt = 1/Fs
        
        density = rho*math.pi/4*d**2
        vel = (tension/density)**.5
        gyradius = d/4
        eps = gyradius**2 * (youngMod*math.pi/4*d**2/(tension*length**2))        
         
    elif (note==45):
        # F4
        length = 0.501
        d = 0.970e-3
        tension = 715

        hammerExponent = 2.482
        hammerLocation = 0.060/length
        hammerMass = 0.00839
        hammerStiffness = 1.076e10
        
        spatialSteps = 50
        dx = length/spatialSteps
        dt = 1/Fs
        
        density = rho*math.pi/4*d**2
        vel = (tension/density)**.5
        gyradius = d/4
        eps = gyradius**2 * (youngMod*math.pi/4*d**2/(tension*length**2))
        
    elif (note == 46):
        #Fd4
        length = 0.475
        d = 0.964e-3
        tension = 711

        hammerExponent = 2.497
        hammerLocation = 0.057/length
        hammerMass = 0.00833
        hammerStiffness = 1.216e10
                
        spatialSteps = 50
        dx = length/spatialSteps
        dt = 1/Fs
        
        density = rho*math.pi/4*d**2
        vel = (tension/density)**.5
        gyradius = d/4
        eps = gyradius**2 * (youngMod*math.pi/4*d**2/(tension*length**2))
        
    elif (note == 47):
        # G4
        length = 0.450
        d = 0.958e-3
        tension = 707

        hammerExponent = 2.512
        hammerLocation = 0.054/length
        hammerMass = 0.00827
        hammerStiffness = 1.374e10
        
        spatialSteps = 50
        dx = length/spatialSteps
        dt = 1/Fs
        
        density = rho*math.pi/4*d**2
        vel = (tension/density)**.5
        gyradius = d/4
        eps = gyradius**2 * (youngMod*math.pi/4*d**2/(tension*length**2))
        
    elif (note == 48):
        #Ab4
        length = 0.426
        d = 0.952e-3
        tension = 704

        hammerExponent = 2.527
        hammerLocation = 0.051/length
        hammerMass = 0.00821
        hammerStiffness = 1.553e10
        
        spatialSteps = 50 
        dx = length/spatialSteps
        dt = 1/Fs        
        
        density = rho*math.pi/4*d**2
        vel = (tension/density)**.5
        gyradius = d/4
        eps = gyradius**2 * (youngMod*math.pi/4*d**2/(tension*length**2))        
    
    elif (note == 49):
        # A4
        length = 0.404
        d = 0.947e-3
        tension = 701

        hammerExponent = 2.543
        hammerLocation = 0.048/length
        hammerMass = 0.00814
        hammerStiffness = 1.755e10
        
        spatialSteps = 50
        dx = length/spatialSteps
        dt = 1/Fs
        
        density = rho*math.pi/4*d**2
        vel = (tension/density)**.5
        gyradius = d/4
        eps = gyradius**2 * (youngMod*math.pi/4*d**2/(tension*length**2))
        
    elif (note == 50):
        #Ad4
        length = 0.383
        d = 0.941e-3
        tension = 699

        hammerExponent = 2.559
        hammerLocation = 0.046/length
        hammerMass = 0.00808
        hammerStiffness = 1.983e10
        
        spatialSteps = 40
        dx = length/spatialSteps
        dt = 1/Fs        
        
        density = rho*math.pi/4*d**2
        vel = (tension/density)**.5
        gyradius = d/4
        eps = gyradius**2 * (youngMod*math.pi/4*d**2/(tension*length**2))
        
    elif (note == 51):
        # B4
        length = 0.363
        d = 0.937e-3
        tension = 697

        hammerExponent = 2.576
        hammerLocation = 0.044/length
        hammerMass = 0.00802
        hammerStiffness = 2.241e10
        
        spatialSteps = 40
        dx = length/spatialSteps
        dt = 1/Fs
        
        density = rho*math.pi/4*d**2
        vel = (tension/density)**.5
        gyradius = d/4
        eps = gyradius**2 * (youngMod*math.pi/4*d**2/(tension*length**2))
        
    elif (note == 52):
        length = 0.344
        d = 0.932e-3
        tension = 696

        hammerExponent = 2.593
        hammerLocation = 0.041/length
        hammerMass = 0.00796
        hammerStiffness = 2.532e10
        
        spatialSteps = 40
        dx = length/spatialSteps
        dt = 1/Fs
        
        density = rho*math.pi/4*d**2
        vel = (tension/density)**.5
        gyradius = d/4
        eps = gyradius**2 * (youngMod*math.pi/4*d**2/(tension*length**2))
    
    else:
        print ("Error: Unknown note")
        
    return length, tension, b1, b3, hammerExponent, hammerLocation, hammerMass, hammerStiffness, hammerSize, hammerVelocity, dx, tmax, Fs, dt, density, eps, vel
    