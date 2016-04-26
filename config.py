import math

#values for the Stanford method C4
#using the table form the Stanford paper
length = 0.63;
c = 329.6;
kap = 1.25;
b1 = 1.1;
b2 = 2.7e-4;

#not from Stanford paper, but should be close to correct!
density = 3.93 / 0.62 / 1000;


dx = length / 50;  # x element size (m)

tmax = 5  # Simulated time [s]
dt = 1 / (96e3);  # Time step [s]

hammerSize = 0.001  # Length of string hit by hammer
hammerVelocity = 1  # Initial velocity of string due to hammer
hammerLocation = 0.12  # Relative distance from the end of the string to hammer striking location
hammerMass = 0.00297     #Hammer mass in kg
hammerStiffness = 4.5e9
hammerExponent = 2.5

# hammerSize = 0.02                   #Length of string hit by hammer
# hammerVelocity = 1                  #Initial velocity of string due to hammer
# hammerLocation = 0.15               #Distance from the end of the string to hammer striking location
# hammerMass = 0.0049                 # hammer mass in kg

bridgePos = hammerLocation / 2;
format = 1;  # DUnno what this means
CHUNK = 5 * 1024  # CHUNK load size
numChannels = 1;  # = Mono
framerate = int(1 / dt);