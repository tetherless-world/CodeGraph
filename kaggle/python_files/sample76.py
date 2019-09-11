#!/usr/bin/env python
# coding: utf-8

# ## Predicting Earthquakes?
# Predicting earthquakes has long been thought to be near-impossible. But same has been said about many other things, I still remember how 15 years ago my IT teacher in high school was saying that speech recognition can never be done reliably by a computer, and today it's more reliable than a human. So who knows, maybe this competition will be a major step in solving another "impossible" problem. Importance of predicting earthquakes is hard to underestimate - selected years of our century claimed hundreds of thousands of causalities. Being able to predict earthquakes could allow us to better protect human life and property.

# In[ ]:


# please ignore this (some peculiarity of Kaggle virtualisation)

# In[ ]:


# standard imports
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd
# found this library when looking at this competiton - it proved worthy! :)
import librosa
import librosa.display

# let's read 10M rown only for now (the dataset is HUGE! over 600M records)
df = pd.read_csv("../input/train.csv", nrows=10_000_000)

# let's rename columns to something easier to type: Signal, Time
df.columns=["s","t"]

# let's identify indices of the "earthquakes"
# (please look at the data description, if this doesn't make sense)
quake_indices = df.index[df.t.diff()>0]

# "Fourier Transformations" - just google that - it's a big, but fruitful topic
# basically this allows us to "see" patterns of "waves" inside time-series data
X = librosa.stft(df.s.values.astype("float").clip(-50,50))
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(18, 6))
ax = librosa.display.specshow(Xdb, sr=1, x_axis="time", y_axis=None)
ax.set_title('Spectrogram of the first 10M points of data provided - white line signifies the earthquake')

# drawing white lines (just one), where earthquake occured
for quake in quake_indices:
    plt.axvline(x=quake, color="white")



# ## Data Provided
# Training data-set for this competition consists of a long time-series of measurements from some sort of an earthquake simulation in a laboratory. As a "test" dataset we are given bits of similar time-series, and the question we have to answer is "when". When the next earthquake (in a lab) is going to happen. The idea is that by analysing those time-series measurements and having some examples of when those lab earthquakes were registered in the past, we can make predictions about the future.

# In[ ]:


# let's take every 200th line (including first), since "train.csv" is huge.
# we store those lines in "./train_every_Nth.csv" for later reading with pandas

# let's create DataFrame from those lines
df_n = pd.read_csv("./train_every_Nth.csv", skiprows=1, names=["s","t"])

# let's index every earthquake
quake_indices = df_n.index[df_n.t.diff()>0]
quake_indices

# creating figure and clipping dataset
plt.figure(figsize=(18, 6))
data = df_n.s.clip(-200,200)

# a simple "data.plot()" would've suffice.. just to make it look a little bit better
for i in range(20):
    data[i::20].plot(alpha=0.05,color="midnightblue")

plt.gca().set_title('Whole dataset - pink dashed lines are earthquakes')

# drawing eartquake lines
for quake in quake_indices:
    plt.axvline(x=quake, linestyle = ":", color="salmon")



# ## Hear it coming?
# When looking at the data in this form it looks suspiciously like sound waves. So my initial guess was to listen to it - to "audiolise" it, just like we're visualising graphical data. It was unknown if the earthquake can be predicted: "by ear". Just like in a dubstep song - the listener can hear the build-up before the "drop", so, perhaps, an earthquake can be felt in a similar fashion.

# In[ ]:


# found these values by investigating tryint to include data around the first "event" 
START = 5_200_000
END = 5_900_000

# taking an "interesting" sample of that data (one which includes an "event")
samp = df.iloc[START:END]
quake_indices = df.index[df.t.diff()>0]

plt.figure(figsize=(18, 6))

ax = plt.gca()
ax.set_title('Part of the raw data just around the earthquake - red line is when the earthquake occured')

# lets plot sample, with event indicated
for i in range(1,10,1):
    (samp.s*i/10).clip(-70,70).plot(alpha=(1-i/10), color="blue")
    plt.axvline(x=quake_indices[0]+1000*i, alpha=(1-i/10), color="red")

# ## Is 44100 Hz enough?
# When interpreting signals as sounds, it's not quite obvious how one should go about it. Perhaps signal coming from experiment itself wasn't audible - for instance, it was infrasound. Either way, one approach was to just treat the data as a digital audio signal at 44100 Hz frequency. 44100 was chosen because it's most common audio resolution (a bit like 720p for video).

# In[ ]:


# we'll be using this library to convert dataframe into sound
from scipy.io import wavfile
# HTML tag to show audio player
from IPython.core.display import HTML


# constant for sampling frequency
AUDIO_RATE = 44100
# amplification constant - so the sound is a bit louder
AMP_CONST = 400
# adding "volume channel" to dataframe
df["v"] = df.s * AMP_CONST

# take a sub-sample from initial dataframe
START = 5_200_000
END = 5_900_000
samp = df.iloc[START:END]

# creating audiowave and writing into a file
wave = (samp.v.values).astype("int16")
wavfile.write("./sound.wav", AUDIO_RATE, wave)

# displaying a player
src = """<audio controls="controls" style="width:600px" >
    <source src="./sound.wav" type="audio/wav" />
      Audio NOT working!</audio>"""
display(HTML(src))

# ## De-noising the signal
# The signal seemed to be full of noise, so one thing to try was to try to remove that noise by using 'sox' tool, which is conveniently provided by Kaggle image. The sound became more "audible", though it was still hard to make sense of what's happening!
# 

# In[ ]:


# filename to use
AUDIO = "./sound"

# this is part where almost no "clicking" happens - we're using it as a guide to define what "noise" is
# creating noise profile before removing
# here we're remoiving the noise

src = """<audio controls="controls" style="width:600px" >
    <source src="./sound_clean.wav" type="audio/wav" />
      Audio NOT working!</audio>"""
display(HTML(src))

# ## Animating the wave
# What's lacking in the previous sample, is the ability to know when the earthquake was registered. In order to have that - we need to have some visualisation.

# In[ ]:


from matplotlib import lines
# Fourier Transformations again
from numpy.fft import rfft
import numpy as np

# setting video rate to 24, for 'smooth' animation
VIDEO_RATE = 24

# setting up sub-plots, first one is for Fourier, second one for Amplitude
fig, ax = plt.subplots(1, 2,figsize=(18,6))

# earthquake locations
quake_indices = samp.index[samp.t.diff()>0]
min_index = samp.index.min()

# sliding line to show which part is playing
def init_blue_line(axi=ax):
    global blue_line
    axi.plot(samp.v, color="skyblue")
    for quake in quake_indices:
        axi.axvline(x=quake, color="red")
    blue_line, = axi.plot([min_index, min_index], [0, 0], color="blue")

    return axi,

def animate_blue_line(num,axi=ax):
    blue_line.set_data([num, num], [+100000, -100000])
    return axi,

# Fourier Transformation to show what's "happening" at that time
def init_rfft(axi=ax):
    global rfft_line
    fourier = rfft(samp.v.loc[min_index:min_index+10000].values)[1:1001]
    fourier = abs(fourier)
    axi.set_xlim(0, 1000)
    axi.set_ylim(-500_000, 3_000_000)
    rfft_line, = axi.plot([],[], color="blue")

def animate_rfft(num, axi=ax):
    fourier = rfft(samp.v.loc[num:num+10000].values)[1:1001]
    fourier = abs(fourier)
    rfft_line.set_data(np.arange(0,len(fourier)), fourier)

def init_all():
    init_rfft(ax[0])
    init_blue_line(ax[1])

# this is just how matplotlib animation works - we need one function to update the whole frame
def animate_all(num):
    animate_rfft(num, ax[0])
    animate_blue_line(num, ax[1])

# In[ ]:


# imports and setting for animation
from matplotlib import animation, rc
rc('animation', html='jshtml')

# defining frames
frames = np.arange(samp.index.min(), samp.index.max(), AUDIO_RATE/VIDEO_RATE)#[:24]

# creating animation
anim = animation.FuncAnimation(fig, animate_all, init_func=init_all, frames=frames, interval=1000/VIDEO_RATE, blit=False)
anim

# ## Putting two together
# Now all that's left is to combine sound with video. This is not a particularly fast or easy process in a Kaggle notebook; be sure you are ready to wait if you want to re-run it. Also, some libraries are missing from Kaggle image - so the next cell does a little "fix".

# In[ ]:


# installing video libraries
# this might take several minutes

# In[ ]:


# importing movie library
import moviepy.editor as mpe

# creating animation with sound - this takes literally forever! (expect 5 mins or more)
fig, ax = plt.subplots(1, 2,figsize=(18,6))
anim = animation.FuncAnimation(fig, animate_all, init_func=init_all, frames=frames, interval=1000/VIDEO_RATE, blit=False)
anim.save('animation.gif', fps=VIDEO_RATE)
my_clip = mpe.VideoFileClip('./animation.gif')
audio_background = mpe.AudioFileClip('./sound.wav')
final_audio = mpe.CompositeAudioClip([audio_background])
# this looks like a hack and it is.. it should be that "my_clip.fps==VIDEO_RATE"
# but for some reason it's not. Matplotlib isn't the best video saver..
# or moviepy isn't the best movie reader.. it's one or the other.
# basically when we're writing, we're setting fps to 24..
# but when we're reading, it's 25 fps all of a sudden
# Either this "anim.save('animation.gif', fps=VIDEO)" doesn't respect the FPS
# or this "mpe.VideoFileClip('./animation.gif')" doesn't read the file properly
# doesn't relly matter who... doing "speedx" does a "quick fix!"
final_clip = my_clip.speedx(VIDEO_RATE/my_clip.fps).set_audio(final_audio)
final_clip.write_videofile("./animation_with_sound.mp4")

# finally display produced animation
HTML("""
<div>
<video width="760" height="440" controls>
  <source src="./animation_with_sound.mp4" type="video/mp4">
</video>
</div>""")

# ## Compare Laboratory Earthquake to Dubstep Drop
# To make sure we did everything correctly, let's do the exact same procedure for a piece of music - "Nero & Skrillex - Promises". The used sample is under 15 seconds, so I hope "fair use" applies!

# In[ ]:


x = mpe.AudioFileClip('../input/skrillex-codecs/promises_short.wav').to_soundarray()[:,0]

samp = pd.DataFrame()
samp["s"] = x
samp["v"] = x*7_000

DROP = 8.314 * AUDIO_RATE
quake_indices = [DROP]
min_index = samp.index.min()
fig, ax = plt.subplots(1, 2,figsize=(18,6))

# In[ ]:


frames = np.arange(samp.index.min(), samp.index.max(), AUDIO_RATE/VIDEO_RATE)

anim = animation.FuncAnimation(fig, animate_all, init_func=init_all, frames=frames, interval=1000/VIDEO_RATE, blit=False)
anim.save('animation_drop.gif')

my_clip = mpe.VideoFileClip('./animation_drop.gif')
audio_background = mpe.AudioFileClip('../input/skrillex-codecs/promises_short.wav')
final_audio = mpe.CompositeAudioClip([audio_background])
final_clip = my_clip.speedx(VIDEO_RATE/my_clip.fps).set_audio(final_audio)
final_clip.write_videofile("./animation_with_dubstep.mp4")

HTML("""
<div>
<video width="760" height="440" controls>
  <source src="./animation_with_dubstep.mp4" type="video/mp4">
</video>
</div>""")

# *Thank you for your attention - I know that this Notebook is a bit of a disorganised dump of ideas, but I really just wanted to put something out already! Any feedback greatly appreciated, and perhaps I could improve on it in the future!*
