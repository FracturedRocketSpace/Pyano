import numpy as np
import math
import config as c

#theoretical values
speed = np.sqrt(c.tention/c.density)
f0 = speed/(2*c.length);
inharmonicity = math.pi**2*c.elasticModulus*c.crossArea*c.gyration**2/(c.tention*c.length**2);

#inharmonic partial frequencies
n = 1;
fn = n*f0*np.sqrt(1+inharmonicity*n**2);


#simulate strings

#plot/animate results: string animation, frequency spectrum

#output sounds

#UI for playing music