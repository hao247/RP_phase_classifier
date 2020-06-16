# from sklearn.ensemble import RandomForestClassifier
import os
from tensorflow.keras.models import load_model
from joblib import load

nn_classifier = load_model(os.getcwd() + '/models/nn_classifier.h5')
rf_classifier = load(os.getcwd() + '/models/rf_classifier.joblib')
