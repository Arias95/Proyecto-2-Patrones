
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from dataset import get_set
from model import getModel
from plots import plot
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


model_name = 'model/model_train.hdf5'

def train(audio_path,plot_matrix = False):

    x_data, y_data=get_set(26,9,audio_path)
    x_data = keras.preprocessing.sequence.pad_sequences(x_data, maxlen=100)

    x_train, x_test, Y_train, Y_test = train_test_split(x_data, y_data,test_size=0.1, random_state=42)


    y_train = keras.utils.to_categorical(Y_train, 16)
    y_test = keras.utils.to_categorical(Y_test, 16)

    model = getModel((x_train.shape[1],x_train.shape[2]), y_train.shape[1])

    history = model.fit(x_train, y_train,batch_size=10,epochs=137, verbose=1,validation_data=(x_test, y_test))
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save(model_name)

    if plot_matrix:
        plot(x_test, Y_test,model_name)


if __name__=='__main__':
    train('audio/normalized_data',plot_matrix=True)