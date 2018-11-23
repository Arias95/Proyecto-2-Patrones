from __future__ import print_function

import itertools
import numpy as np
import sys
import keras
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from dataset import get_set



classes = ['cero','uno','dos','tres','cuatro','cinco','seis','siete','ocho','nueve','diez','once','doce','trece','catorce', 'quince']

def plot_confusion_matrix(cm, title):

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def plot(x_test, y_test, imported_model):
    model=load_model(imported_model)

    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=100)

    y_predict = model.predict_classes (x_test, verbose=0)
    
    plt.figure()
    cm = confusion_matrix(y_test,y_predict)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(cm, title='Confusion matrix')
    plt.show()



def semi_plot():
    model=load_model('model/model7.hdf5')

    x_test, y_test=get_set(13,9,'audio/test_normalized_data')
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=100)

    y_predict = model.predict_classes (x_test, verbose=0)
    
    plt.figure()
    cm = confusion_matrix(y_test,y_predict)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(cm, title='Confusion matrix')
    plt.show()



if __name__== "__main__":
        semi_plot()