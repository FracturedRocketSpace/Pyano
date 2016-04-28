import math

format = 1;  # DUnno what this means
minCHUNK = 1024  # CHUNK load size
numChannels = 1;  # = Mono
numProcesses = 8;
numTasks = None; # Set to non-zero value to refresh workers periodically
bridgePos = -3; #Last string segment before end
synthMode = 1; # Decreases norm to create another sound
spectrum = False;   #Toggle to turn on spectrum calculation and plotting after playing the sound

def selectParameters(note):    
    youngMod = 2e11
    rho = 7850
    tmax = 2.5
    Fs =int(64e3)
    hammerSize = 0.01
    hammerVelocity = 1
    
    if (note==28):
        #C3
        length = 1.259
        d = 1.063e-3
        tension = 759
            
        hammerExponent = 2.312 
        hammerLocation = 0.151/length
        hammerMass = 0.00945
        hammerStiffness = 1.347e9
        
        spatialSteps=50
        
    elif (note==29):
        #Cd3
        length = 1.192
        d = 1.061e-3
        tension = 762
            
        hammerExponent = 2.318 
        hammerLocation = 0.143/length
        hammerMass = 0.00939
        hammerStiffness = 1.522e9
    
        spatialSteps=50   
    
    elif (note==30):
        #D3
        length = 1.129
        d = 1.059e-3
        tension = 764
            
        hammerExponent = 2.325 
        hammerLocation = 0.136/length
        hammerMass = 0.00933
        hammerStiffness = 1.720e9
        
        spatialSteps=50
        
    elif (note==31):
        #Dd3
        length = 1.070
        d = 1.057e-3
        tension = 766
            
        hammerExponent = 2.332 
        hammerLocation = 0.128/length
        hammerMass = 0.00927
        hammerStiffness = 1.943e9

        spatialSteps=50
        
    elif (note==32):
        #E3
        length = 1.013
        d = 1.053e-3
        tension = 767
            
        hammerExponent = 2.339 
        hammerLocation = 0.122/length
        hammerMass = 0.00920
        hammerStiffness = 2.196e9

        spatialSteps=50
        
    elif (note==33):
        #F3
        length = 0.960
        d = 1.049e-3
        tension = 766
            
        hammerExponent = 2.347 
        hammerLocation = 0.115/length
        hammerMass = 0.00914
        hammerStiffness = 2.481e9

        spatialSteps=50
    
    elif (note==34):
        #Fd3
        length = 0.909
        d = 1.045e-3
        tension = 765
            
        hammerExponent = 2.356 
        hammerLocation = 0.109/length
        hammerMass = 0.00908
        hammerStiffness = 2.804e9

        spatialSteps=50
    
    elif (note==35):
        #Fd3
        length = 0.861
        d = 1.039e-3
        tension = 763
            
        hammerExponent = 2.365 
        hammerLocation = 0.103/length
        hammerMass = 0.00902
        hammerStiffness = 3.169e9

        spatialSteps=50
    
    elif (note == 36):
        #Ab3
        length = 0.816
        d = 1.033e-3
        tension = 759

        hammerExponent = 2.375
        hammerLocation = 0.098/length
        hammerMass = 8.96e-3;
        hammerStiffness = 3.581e9
        
        spatialSteps = 50 
    elif (note == 37):
        #A3
        length = 0.773
        d = 1.027e-3
        tension = 755

        hammerExponent = 2.385
        hammerLocation = 0.093/length
        hammerMass = 8.89e-3;
        hammerStiffness = 4.047e9
        
        spatialSteps = 50 
    elif (note == 38):
        #Ad3
        length = 0.732
        d = 1.020e-3
        tension = 751

        hammerExponent = 2.395
        hammerLocation = 0.088/length
        hammerMass = 8.83e-3;
        hammerStiffness = 4.573e9
        
        spatialSteps = 50 
    elif (note == 39):
        #B3
        length = 0.694
        d = 1.013e-3
        tension = 746

        hammerExponent = 2.406
        hammerLocation = 0.083/length
        hammerMass = 8.77e-3;
        hammerStiffness = 5.168e9
        
        spatialSteps = 50 
    elif (note==40):
        # C4
        length = 0.657
        d = 1.006e-3
        tension = 741
            
        hammerExponent = 2.418 #2.8
        hammerLocation = 0.079/length
        hammerMass = 0.00871
        hammerStiffness = 5.840e9 #4.84e9
        
        spatialSteps = 50
              
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
                
    elif (note == 52):
        length = 0.344
        d = 0.932e-3
        tension = 696

        hammerExponent = 2.593
        hammerLocation = 0.041/length
        hammerMass = 0.00796
        hammerStiffness = 2.532e10
        
        spatialSteps = 40
           
    else:
        print ("Error: Unknown note")
        
    #other parameter calculations
    dx = length/spatialSteps
    dt = 1/Fs
    
    density = rho*math.pi/4*d**2
    vel = (tension/density)**.5
    gyradius = d/4
    eps = gyradius**2 * (youngMod*math.pi/4*d**2/(tension*length**2))
    #kap = eps * (vel**2) * (length**2)

    #loss parameters
    f0 = vel/(2*length)
    b1 = 4.4e-3 * f0 - 4e-2;
    #b2 = 1e-6 * f0 + 1e-5;
    b3 = 6.25e-9;
    
    print("Theoretical f0 = " + str(f0));
    
    #dt_max = dx**2 * (-4 * b2 + (16 * b2**2 + 4*(vel**2 * dx**2 + 4 * kap**2))**(1/2))/(2*(vel**2*dx**2+4*kap**2))
    #print("dt: " + str(dt));    
    #print("max dt: " + str(dt_max));
        
    return length, tension, b1, b3, hammerExponent, hammerLocation, hammerMass, hammerStiffness, hammerSize, hammerVelocity, dx, tmax, Fs, dt, density, eps, vel
    