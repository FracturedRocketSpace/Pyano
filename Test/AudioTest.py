import pyaudio
import numpy as np
import sys
import timeit


# Settings
CHUNK = 5*1024
numChannels = 1;
framerate = 44100;
length = 5; # Num seconds
frequency = 2109.89; # Hz

# Create time
t = np.linspace(0, length, num=length*framerate)

# Initiate the PyAudio instance
p = pyaudio.PyAudio()
# Open stream to audio device
# Format: Array type. Int32 or float32 for example. 1 = float32?
# Channels. Number of channels. 1=mono, 2=stereo
# Rate: The sampling rate
# Output: True of course as we want output
stream = p.open(format=p.get_format_from_width(1),
                channels=numChannels,
                rate=framerate,
                output=True)
# Audio output
audio = "";
print('One time play')
audio = np.array(np.sin( (2*np.pi)*frequency * t )*127 + 128, dtype=np.int8).view('c')
start = timeit.default_timer()

stream.write(audio)
stop = timeit.default_timer()
print("Program ended in  =", int(stop - start), "seconds");

# Audio output V2
print('Chunk loading')
audio = np.array(np.sin( (2*np.pi)*frequency * t )*127 + 128, dtype=np.int8);

start = 0;
end = CHUNK;

startT= timeit.default_timer()
while(end<=framerate*length):
    data = audio[start:end].view('c')
    start=end;
    end=end+CHUNK;
    stream.write(data)

print("Program ended in  =", int(timeit.default_timer() - startT), "seconds");

# Stop the audio output
stream.stop_stream()
stream.close()

p.terminate();