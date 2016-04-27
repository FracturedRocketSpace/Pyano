import math


tension = 670  # String tension [N]
density = 3.93 /1000/ 0.62;  # Density [kg/m]

c = (tension / density) ** (1 / 2)  # Speed of sound
eps = 3.82e-5;  # Stiffness
b1 = 0.5;  # First damping
b3 = 6.25e-9;  # Third damping

length = 0.62;  # String length [m]
dx = 0.62 / 25;  # x element size (m)

tmax = 5  # Simulated time [s]
dt = 1 / (32e3);  # Time step [s]

hammerSize = 0.001  # Length of string hit by hammer
hammerVelocity = 1  # Initial velocity of string due to hammer
hammerLocation = 0.2  # Relative distance from the end of the string to hammer striking location
hammerMass = 0.00297     #Hammer mass in kg
hammerStiffness = 4.5e9
hammerExponent = 2.5

bridgePos = hammerLocation / 2;
format = 1;  # DUnno what this means
CHUNK = 2 * 1024 - 1  # CHUNK load size
numChannels = 1;  # = Mono
framerate = int(1 / dt);

norm = 1e-5;