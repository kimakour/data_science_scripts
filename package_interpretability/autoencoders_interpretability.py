import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from keras.models import Model, load_model, clone_model
from keras.layers import Input, Dense, Dropout


def create_one_dimensional_network(model, feature_number):
	"""
	reconstruct a model that has only one input from an autoencoder
	"""

    cloned_model = clone_model(model)
    cloned_model.set_weights(model.get_weights())

    cloned_2 = cloned_model.layers.pop()
    fc = Dense(1, activation="linear")(cloned_model.layers.pop().output)
    model_2 = Model(inputs=cloned_model.input, outputs=fc)

    array_neurons = []
    for i in model.get_weights()[18]:
        array_neurons.append([i[feature_number]])
    array_bias = np.array([model.get_weights()[19][feature_number]])
    array_neurons = np.array(array_neurons)

    model_2.layers[10].set_weights([array_neurons, array_bias])
    return model_2


def create_dictionnary_model(X_train, model):
	"""
	create a dictionnary containing as a key the name of the column to reconstruct
	and as a value a keras model
	"""

    dictionnary = dict()
    for index, value in enumerate(X_train.columns):
        dictionnary[value] = create_one_dimensional_network(model, index)
    return dictionnary


def return_explainer(model, X_train_scaled, frac=0.01):
	"""
	Create a Kernel explainer for the reconstruction of each feature
	"""
    def function(X):

        return model.predict(X)

    explainer = shap.KernelExplainer(
        function, X_train_scaled.sample(frac=frac)
    )
    return explainer


def return_shap_values_for_test(explainer, X_test, frac=0.1):

    shap_values = explainer.shap_values(X_test.sample(frac=frac))
    return shap_values


def store_shap_values_per_feature(feature_name, shap_values):
    string = str(feature_name) + "_shap_values.npy"
    np.save(string, shap_values[0])


def compute_all_shap_values(X_train_scaled, X_test_scaled, model):

    dictionnary = create_dictionnary_model(X_train_scaled, model)
    for i, j in dictionnary.items():
        explainer = return_explainer(j, X_train_scaled)
        shap_values = return_shap_values_for_test(explainer, X_test_scaled)
        store_shap_values_per_feature(i, shap_values)
