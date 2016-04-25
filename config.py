import math

tention = 750                       #String tention [N]
elasticModulus = 2e11               #Young's modulus [N/m^2]
diameter = 1e-3                     #String diameter [m]
crossArea = math.pi/4*diameter**2   #Cross sectional area of a string [m^2]
gyration = diameter/4               #Radius of gyration for a cilindrical string is half the radius [m]
density = 0.018                     #linear density [kg/m]

length = 1                          #String length [m]