import math

tension = 750                       #String tension [N]
density =0.460/0.09;
c = (tension/density)**(1/2)
eps = 8.67e-4;
b1 = 0.5;
b3 = 2.6e-10;


length = 0.09;                         #String length [m]
dx = 0.09/64;                        #x element size (m)

tmax = 1
dt = 1/(96e3);

hammerSize = 0.0054/2               #Length of string hit by hammer
hammerVelocity = 1                  #Initial velocity of string due to hammer
hammerLocation = 0.0054             #Distance from the end of the string to hammer striking location

bridgePos = hammerLocation;
format = 1; # DUnno what this means
CHUNK = 5*1024 # CHUNK load size
numChannels = 1; # = Mono
framerate = int(1/dt);
