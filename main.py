import numpy as np
import math
import config as c
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for projection='3d'!
from matplotlib import cm
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math
import scipy.sparse
import timeit
import pyaudio

x = np.arange(-c.dx, c.length + 2 * c.dx, c.dx)
t = np.arange(0, c.tmax + c.dt, c.dt)

# Initiate arrays for deviation and velocity
dev = np.zeros([len(x), len(t)])
vel = np.zeros([len(x), len(t)])

# Initial conditino: Strike string with hammer
# Gaussian hammer
mean = c.hammerLocation;
variance = c.hammerSize;
sigma = math.sqrt(variance);
vel[:, 0] = c.hammerVelocity * mlab.normpdf(x, mean, sigma);
# Pulse hammer
# vel[int(c.hammerLocation/c.length*len(x) ): int((c.hammerLocation+c.hammerSize)/c.length*len(x)), 0] = c.hammerVelocity;

# force for first iteration
dev[:, 0] += vel[:, 0] * c.dt
dev[:, 1] += vel[:, 0] * c.dt
dev[:, 2] += vel[:, 0] * c.dt

# Create matrices
D = 1 + c.b1 * c.dt + 2 * c.b3 / c.dt;
r = c.c * c.dt / c.dx;
N = len(x)

a1 = (2 - 2 * r ** 2 + c.b3 / c.dt - 6 * c.eps * (N ** 2) * (r ** 2)) / D;
a2 = (-1 + c.b1 * c.dt + 2 * c.b3 / c.dt) / D;
a3 = (r ** 2 * (1 + 4 * c.eps * (N ** 2))) / D;
a4 = (c.b3 / c.dt - c.eps * (N ** 2) * (r ** 2)) / D;
a5 = (- c.b3 / c.dt) / D;
# Padded!
diagonals1 = [a1 * np.ones(N), a3 * np.ones(N - 1), a3 * np.ones(N - 1), a4 * np.ones(N - 2), a4 * np.ones(N - 2)]
diagonals1[0][0] = diagonals1[0][-1] = 0;
diagonals1[1][0] = diagonals1[2][-1] = 0;
diagonals1[3][0] = diagonals1[4][-1] = 0;
A1 = scipy.sparse.diags(diagonals1, [0, 1, -1, 2, -2], format="csr")

diagonals2 = [a2 * np.ones(N), a5 * np.ones(N - 1), a5 * np.ones(N - 1)];
diagonals2[0][0] = diagonals2[0][-1] = 0;
diagonals2[1][0] = diagonals2[2][-1] = 0;
A2 = scipy.sparse.diags(diagonals2, [0, 1, -1], format="csr");

diagonals3 = [a5 * np.ones(N)]
diagonals3[0][0] = diagonals3[0][-1] = 0;
A3 = scipy.sparse.diags(diagonals3, [0], format="csr");

# THE ITERATORN MWHAUHAHAHAH
start = timeit.default_timer()
for i in range(3, len(t)):
    dev[:, i] = A1.dot(dev[:, i - 1]) + A2.dot(dev[:, i - 2]) + A3.dot(dev[:, i - 3]);
    # end zero
    dev[1, i] = 0;
    dev[-2, i] = 0;
    # 2nd
    dev[0, i] = -dev[2, i];
    dev[-1, i] = -dev[-3, i];
    print('Now at ', i + 1, 'of the ', len(t));
print("Program ended in  =", int(timeit.default_timer() - start), "seconds");

# Get sound output
audio = dev[int(c.bridgePos / c.length * len(x)), :];
print(len(audio))
# Normalize and convert
norm = max(abs(audio));
audio = audio / norm;
audio_out = np.array(audio * 127 + 128, dtype=np.int8).view('c');
# Init sound
p = pyaudio.PyAudio()
# Open stream to audio device
# Format: Array type. Int32 or float32 for example. 1 = float32?
# Channels. Number of channels. 1=mono, 2=stereo
# Rate: The sampling rate
# Output: True of course as we want output
stream = p.open(format=p.get_format_from_width(c.format),
                channels=c.numChannels,
                rate=c.framerate,
                output=True)

# output sounds
start = timeit.default_timer()
stream.write(audio_out)
print("Program ended in  =", int(timeit.default_timer() - start), "seconds");

# Stop the audio output
stream.stop_stream()
stream.close()

p.terminate();

# plot/animate results: string animation, frequency spectrum
X, T = np.meshgrid(x, t)

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(X,T ,dev.T, rstride=10, cstride=10,cmap=cm.coolwarm, linewidth=0, antialiased=False)
# ax.view_init(90, 90); # Top view
# plt.xlabel("Position")
# plt.ylabel("Time")
# plt.title("Implicit method")
# fig.colorbar(surf, shrink=0.5, aspect=5)
# Hide z-axis
# ax.w_zaxis.line.set_lw(0.)
# ax.set_zticks([])

plt.figure()
plt.plot(t, audio * 127)

plt.show()
