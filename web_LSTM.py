from flask import Flask
from flask import url_for, render_template, request
import tensorflow
import numpy as np # Ini buat array
#import cv2 # Ini buat gambar
import matplotlib.pyplot as plt
#%matplotlib inline
import librosa # Library untuk membuat MFCC
import pydub # Library untuk membaca mp3/wav
from statistics import mode


app = Flask(__name__)

kamus = {0: "Purr", 1: "Mating", 2:"Meow", 3:"Howl"}

# Ini fungsi buat mbaca mp3
def read(f, normalized=False):
    a = pydub.AudioSegment.from_mp3(f)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return a.frame_rate, np.float32(y) / 2**15
    else:
        return a.frame_rate, y
        

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#===============================================================
# Ini Otak CNNnya
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Bidirectional, TimeDistributed, Conv1D

model = Sequential()

#model.add(Embedding(kosakata, embed_fitur, input_length=x.shape[1], input_shape=x.shape[1:]))
model.add(Conv1D(8, 15, input_shape= (882000, 2), strides=15, activation='relu'))
model.add(Conv1D(8, 15, strides=15, activation='relu'))
model.add(Conv1D(8, 15, strides=15, activation='relu'))
model.add(Conv1D(8, 15, strides=15, activation='relu'))
model.add(LSTM(8, return_sequences=True))
model.add(LSTM(8))
model.add(Dense(16, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.load_weights("C:\\Users\\LAB DATA PC 14\\Documents\\SkripsiDila\\static\\8-8L8888oke.h5")    

@app.route("/")
def hello_world():
    return render_template('halo.html')

@app.route('/ini_apa_upload', methods = ['GET', 'POST'])
def ini_apa_upload():
   if request.method == 'POST':
      f = request.files['file']
      path = './static/suara' + url_for('hello_world') + '_' + f.filename
      f.save(path)
      
      print (path)
      
      sr_kucing, x_kucing = read(path)
      
      x_kucing0 = x_kucing.astype(float)
      panjang = len(x_kucing0)//sr_kucing
      test=[]
      for x in range(0, panjang-20, 20):
      
        test.append(x_kucing0[x*sr_kucing:(x+20)*sr_kucing])
        print("aaaaaaaaaaa ", test[0].shape)

      test = np.asarray(test)
      print("bbbbbbbbbbbb", test.shape)
      hasil = model.predict(test)
      hasil = [np.argmax(_) for _ in hasil]
      
      #Cari yang muncul paling banyak
      jawaban=mode(hasil)
      kalimat = ""
      for x in hasil :
        kalimat = kalimat+kamus[x]
      
      return ("Suara ini masuk ke kategori "+kamus[jawaban])#+str(hasil)+kalimat)
      #