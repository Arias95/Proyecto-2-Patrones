#!/bin/python3
import numpy as np
import keras
import sys
import matplotlib.pyplot as plt
from keras.models import load_model
from scipy.io import wavfile as wav
from plots import plot_confusion_matrix
from dataset import input_vector 
from sklearn.metrics import confusion_matrix



Classes = ['cero','uno','dos','tres','cuatro','cinco','seis','siete','ocho','nueve','diez','once','doce','trece','catorce', 'quince']

def predictNumber(audioPath):
    model=load_model('model/model7.hdf5')

    ##procesar audio, o lo que sea y meter en audio
    fs, audio = wav.read(audioPath) ## tiene que ser normalizado
    X = input_vector(audio, fs, 13, 9)

    
    X=X.reshape(1,X.shape[0],X.shape[1])
    X = keras.preprocessing.sequence.pad_sequences(X, maxlen=100)

    number=model.predict(X)
    #print("System Output: " + str(Classes[np.argmax(number)]))
    return  str(Classes[np.argmax(number)])




if __name__=='__main__':
    audio=sys.argv[1]
    number = predictNumber(audio)
    print("System Output: " + number)