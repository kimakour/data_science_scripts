import random

import pandas as pd
import numpy as np

from keras.models import Model, clone_model
from keras.layers import Dense

import plotly.graph_objects as go
import shap


def create_one_dimensional_network(
    model, feature_number, final_activation="linear"
):
    """ 
    create a one dimensional model based on the feature we want to reconstruct
    It keeps the neuron related to this feature only in the last layer
    Arguments:
    model : keras model 
    feature_number : the feature order number ; from 0 to number_features-1
    final_activation = depends on the activation of the last layer, default = "linear"
    Returns: 
    keras model 
    """
    cloned_model = clone_model(model)
    cloned_model.set_weights(model.get_weights())
    len_weights = len(model.get_weights())
    cloned_2 = cloned_model.layers.pop()
    fc = Dense(1, activation=final_activation)(
        cloned_model.layers.pop().output
    )
    model_2 = Model(inputs=cloned_model.input, outputs=fc)

    array_neurons = []
    for i in model.get_weights()[len_weights - 2]:
        array_neurons.append([i[feature_number]])
    array_bias = np.array(
        [model.get_weights()[len_weights - 1][feature_number]]
    )
    array_neurons = np.array(array_neurons)

    model_2.layers[int(len_weights / 2)].set_weights(
        [array_neurons, array_bias]
    )
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


def spot_indexes_anomalies(X_true, autoencoder, threshold):
    """
    Creates a dictionnary containing the reconstruction of X_true and the indexes
    of anomalies in the data set
    Arguments:
    X_true : the data set we want to reconstruct
    autoencoder : the autoencoder , a keras model 
    threshold : the minimum value of MSE to differentiate between anomalies and normal instances
    Returnes :
    A dictionnary that has as a key :
    "X_pred" : the predictions
    "anomaly_indexes" : list of indexes of anomalies

    """
    dictionnary_anomaly = dict()
    X_pred = pd.DataFrame(autoencoder.predict(X_true), columns=X_true.columns)
    X_pred.index = X_true.index
    anomaly_indexes = X_pred[
        np.mean(np.power(X_true - X_pred, 2), axis=1) > threshold
    ].index
    dictionnary_anomaly["X_pred"] = X_pred
    dictionnary_anomaly["anomaly_indexes"] = anomaly_indexes
    return dictionnary_anomaly


def explain_anomaly(
    X_true,
    X_pred,
    loc_anomaly,
    threshold,
    autoencoder,
    frac=0.01,
    final_activation="linear",
):
    """
    returns a dictionnary containing the features that led to badly reconstruct the input
    ( see research paper )
    Arguments : 
    X_true : real input
    X_pred : prediction 
    loc_anomaly : index of the anomaly 
    threshold : the minimum value of MSE to differentiate between anomalies and normal instances
    autoencdoer: keras model
    frac : the fraction of the X_train to use to construct the kernel explainer
    final_activation : final activation of the keras model (last layer )
    """
    dictionnary_explanations = dict()
    list_columns = X_true.columns

    for num_column, column in enumerate(list_columns):

        if (
            X_pred.loc[loc_anomaly, column] - X_true.loc[loc_anomaly, column]
        ) ** 2 > threshold:
            dictionnary_explanations[column] = list()
            model_one_d = create_one_dimensional_network(
                autoencoder, num_column, final_activation=final_activation
            )
            explainer = shap.KernelExplainer(
                model_one_d.predict, X_true.sample(frac=frac)
            )
            shap_values = explainer.shap_values(X_true.loc[loc_anomaly, :])

            if (
                X_pred.loc[loc_anomaly, column]
                > X_true.loc[loc_anomaly, column]
            ):
                for num_feature, value in enumerate(shap_values[0]):
                    if (round(value, 2) > 0) and (num_feature != num_column):
                        dictionnary_explanations[column].append(
                            dict.fromkeys(
                                {list_columns[num_feature]}, round(value, 2)
                            )
                        )
            elif (
                X_pred.loc[loc_anomaly, column]
                < X_true.loc[loc_anomaly, column]
            ):
                for num_feature, value in enumerate(shap_values[0]):
                    if (round(value, 2) < 0) and (num_feature != num_column):
                        dictionnary_explanations[column].append(
                            dict.fromkeys(
                                {list_columns[num_feature]}, round(value, 2)
                            )
                        )
    dictionnary_explanations_non_empty = {
        k: v for k, v in dictionnary_explanations.items() if len(v) > 0
    }
    return dictionnary_explanations_non_empty


def plot_anomaly_importance(X_train, dictionnary, loc_anomaly):
    """
    plot the features that are important in constructing badly each feature
    """
    plot = []
    for key in dictionnary.keys():
        for i in dictionnary[key]:
            name_value = "{} = {}".format(
                list(i.keys())[0],
                round(X_train.loc[loc_anomaly, list(i.keys())[0]], 2),
            )
            trace = go.Bar(name=name_value, x=[key], y=[list(i.values())[0]])
            plot.append(trace)
    fig = go.Figure(plot)
    fig.show()


def choose_anomaly_randomly(
    X_train,
    X_train_scaled,
    autoencoder,
    threshold,
    frac=0.01,
    final_activation="linear",
):
    """
    Example of an anomaly explanation 
    """

    dictionnary_anomaly = spot_indexes_anomalies(
        X_train_scaled, autoencoder, threshold
    )
    loc_anomaly = random.choice(dictionnary_anomaly["anomaly_indexes"])
    dictionnary_explanation = explain_anomaly(
        X_train_scaled,
        dictionnary_anomaly["X_pred"],
        loc_anomaly,
        threshold,
        autoencoder,
        frac=frac,
        final_activation=final_activation,
    )
    plot_anomaly_importance(X_train, dictionnary_explanation, loc_anomaly)
