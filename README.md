# Proyecto 2 Patrones
Spoken number recognition

Outputs a number from 0 to 15 according to the input
spoken number (in spanish).

## Made by
* Francisco Alvarado Ferllini
* Andrés Arias Vargas
* Óscar Ulate Alpizar
* Pablo Rodríguez Quesada

## Course information
Costa Rica Institute of Technology

Introduction to Pattern Recognition

Professor: Dr. Pablo Alvarado Moya

## How to use

First run the `load_audios.py` script in order to load the test
data.
``
python3 src/load_audios.py
``

Then, run the `train.py` script to begin the training process.
``
python3 src/train.py
``

After the training process is finished, run the `predict.py`
script to run the tests.
``
python3 src/predict.py
``

Finally, run the `recognition.py` script to record a new spoken number
and predict the value using the trained model.
``
python3 src/recognition.py
``

The `plots.py` script will give you the confusion matrix for the 
trained model.
``
python3 src/plots.py
``
