import math

tension = 750                       #String tension [N]
elasticModulus = 2e11               #Young's modulus [N/m^2]
diameter = 1e-3                     #String diameter [m]
crossArea = math.pi/4*diameter**2   #Cross sectional area of a string [m^2]
gyration = diameter/4               #Radius of gyration for a cilindrical string is half the radius [m]
density = 8050*crossArea                   #linear density [kg/m]

length = 1                          #String length [m]
dx = 0.001                          #x element size (m)

tmax = 0.5
dt = 0.0001

hammerSize = 0.02                   #Length of string hit by hammer
hammerVelocity = 1                  #Initial velocity of string due to hammer
hammerLocation = 0.15               #Distance from the end of the string to hammer striking location
hammerMass = 0.0049                 # hammer mass in kg
