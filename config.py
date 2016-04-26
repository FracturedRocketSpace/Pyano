import math

tension = 750  # String tension [N]
density = 0.460 / 0.09;  # Densiuty [kg/m]
c = (tension / density) ** (1 / 2)  # Speed of sound
eps = 8.67e-4;  # Stiffness
b1 = 0.5;  # First damping
b3 = 2.6e-10;  # Third damping

length = 0.09;  # String length [m]
dx = 0.09 / 64;  # x element size (m)

tmax = 5  # Simulated time [s]
dt = 1 / (96e3);  # Time step [s]

hammerSize = 0.0054 / 2  # Length of string hit by hammer
hammerVelocity = 1  # Initial velocity of string due to hammer
hammerLocation = 0.0054  # Distance from the end of the string to hammer striking location

bridgePos = hammerLocation / 2;
format = 1;  # DUnno what this means
CHUNK = 5 * 1024  # CHUNK load size
numChannels = 1;  # = Mono
framerate = int(1 / dt);