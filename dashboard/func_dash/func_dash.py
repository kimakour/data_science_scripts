import pandas as pd
import numpy as np
import dash_core_components as dcc
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from numpy import mean
from numpy import absolute
from scipy.integrate import simps
import shap
from sklearn.ensemble import GradientBoostingRegressor


def REC(y_true, y_pred, min_Range, max_Range, interval_size):
    """
    Compte REC values given the y_true and y_pred 

    Arguments:
    - y_true : numpy array containing the values of y_test
    - y_pred : numpy array containing the values of y_predicted
    - min_range : minimum error 
    - max_range : maximum error 
    - interval_size : path used for compute each step of REC

    Returns : 
    - epsilon : Error values for each step
    - accuracy : percentage of samples that are below the max error in each step
    - auc : area under curve for the REC 
    """
    accuracy = []
    epsilon = np.arange(min_Range, max_Range, interval_size)
    for i in range(len(epsilon)):
        count = 0.0
        for j in range(len(y_true)):
            if np.abs(y_true[j] - y_pred[j]) < epsilon[i]:
                count = count + 1
        accuracy.append(count / len(y_true))
    auc = simps(accuracy, epsilon) / max_Range
    return epsilon, accuracy, auc


def intervals(x):
    """
    Defining intervals for the error
    """
    if x < 0.1:
        return "absolute error < 0.1 "
    elif x >= 0.1 and x < 0.5:
        return "0.1 < absolute error < 0.5 "
    elif x >= 0.5 and x < 1:
        return "0.5 < absolute error < 1 "
    else:
        return "absolute error > 1 "


def create_dcc(possible_values, name, default_value):
    """
    Create an html dcc component 

    Arguments:
    - possible_values: possible values for label and value of each option
    - name: id name of the dcc component 
    - default_value: default value of the dcc in the initialization of the app

    Returns:
    - dcc component
    """
    return dcc.Dropdown(
        id=name,
        options=[{"label": c, "value": c} for c in possible_values],
        value=default_value,
    )


def create_slider(name, mini, maxi, marks):
    """
    Create an html slider

    Arguments:
    - name : id name of the slider component
    - mini : minimum value
    - maxi : maximimum value 
    - marks : marks for each value ; example {0:"None", 1:"Output"}

    Returns:
    - slider component
    """
    return dcc.Slider(
        id=name,
        min=mini,
        max=maxi,
        marks=marks,
        value=0,
        className="pretty_container",
    )


def create_radio_shape(name):
    """
    Create an html radio component to choose the way to plot the distribution
    4 possibles modes: violin plot, box plot, histogram, rug plot

    Arguments:
    - name : id name of the radio component

    Returns:
    - radio component
    """
    return dcc.RadioItems(
        id=name,
        options=[
            {"label": "violin", "value": "violin"},
            {"label": "box", "value": "box"},
            {"label": "hist", "value": "histogram"},
            {"label": "rug", "value": "rug"},
        ],
        value="histogram",
        className="pretty_container",
    )


def re_order_dataset(df, is_linear_regression=True):
    """
    Re-order the dataset in a way where the Width is the target  variable

    Arguments:
    - df : pandas dataframe
    - is_linear_regression : boolean to creae dummy variables with deleting the last dummy
    default value: True

    Returns:
    - encoded_df: one hot encoded dataframe 
    """
    encoded_df = pd.get_dummies(df, drop_first=is_linear_regression)
    encoded_df = encoded_df[
        list(set(encoded_df.columns.tolist()) - set(["Width"])) + ["Width"]
    ]
    return encoded_df


def create_dataframe_results(X, y, X_train, X_test, y_train, y_test, kfolds=5):
    """
    Performs a linear regression and returns general regression metrics and y predicted.

    Arguments:
    - X: the whole dataset for the predictive features 
    - y: the whole dataset for the target variable
    - X_train: train set for the predicttive features
    - X_test: test set for the predictive features
    - y_train: train set for the target variable
    - y_test: test set for the target variable
    - kfolds: number of folds for cross-validation; default: 5

    Returns:
    tuple containing: 
    - dataframe_results: pandas dataframe with following columns: "mean crossval MSE", 
    "R2 score", "MSE", "MAE"
    - dataframe_absolute_error: pandas dataframe containing to which interval the error belongs
    - y_pred : predictions for the y test.

    """

    cv = KFold(n_splits=kfolds, shuffle=True, random_state=1)
    pipeline = Pipeline(
        steps=[("normalize", MinMaxScaler()), ("model", LinearRegression())]
    )
    scores = cross_val_score(
        pipeline, X, y, scoring="neg_mean_squared_error", cv=cv, n_jobs=-1
    )
    scores = absolute(scores)
    s_mean = mean(scores)

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    dataframe_results = pd.DataFrame(
        [[s_mean, r2, mse, mae]],
        columns=["mean crossval MSE", "R2 score", "MSE", "MAE"],
    )

    dataframe_absolute_error = pd.DataFrame(np.abs(y_test - y_pred))
    dataframe_absolute_error["intervals"] = dataframe_absolute_error.apply(
        lambda x: intervals(x["Width"]), axis=1
    )

    return dataframe_results, dataframe_absolute_error, y_pred


def create_values_pie_chart(dataframe_absolute_error):
    """
    Creates lists containing the name of labels and values for each one of them for the intervals 
    of errors 

    Arguments:
    - dataframe_absolute_error: dataframe containing the affiliation of each error to 
    the error interval

    Returns:
    lables: list of strings containing the name of labels
    values: list of values related to the labels 

    """
    labels = [
        "absolute error < 0.1 ",
        "0.1 < absolute error < 0.5 ",
        "0.5 < absolute error < 1 ",
        "absolute error > 1 ",
    ]
    values = []
    for i in labels:
        values.append(
            len(
                dataframe_absolute_error[
                    dataframe_absolute_error["intervals"] == i
                ]
            )
        )
    return labels, values


def create_REC_plot(y_test, y_predicted):
    """
    Creates dataframes necessary for the REC plot

    Arguments:
    - y_test: test set for the target variable
    - y_predicted : predicted target variable

    Returns:
    - dataframe_error : dataframe containing y_test, y_pred and the error
    - df_REC : dataframe containing the deviance and accuracy 
    - title_REC: title for the REC plot
    """
    error = pd.DataFrame(
        y_test["y_test"] - y_predicted["y_pred"], columns=["error"]
    )
    error.index = y_test.index
    dataframe_error = pd.concat([error, y_test, y_predicted], 1)
    deviance, accuracy, auc = REC(
        dataframe_error["y_test"].values,
        dataframe_error["y_pred"].values,
        0,
        3,
        0.05,
    )
    d = {"Deviance": deviance, "accuracy": accuracy}
    df_REC = pd.DataFrame(data=d)
    title_REC = "REC curve , auc score=" + str(round(auc, 3))
    return dataframe_error, df_REC, title_REC

def create_explainer(X_train, y_train):
    """
    Creates a Tree explainer for X_train, y_train using shap values and Gradient boosting regressor

    Arguments:
    - X_train: train set for the features 
    - y_train: train set for the target 

    Returns:
    - model_GBT: trained gradient boosting regressor model
    - base_value: mean value of the predictions on the train set
    - explainer: SHAP tree explainer 

    """
    model_GBT = GradientBoostingRegressor().fit(X_train, y_train)
    base_value = str(round(model_GBT.predict(X_train).mean(), 2))
    explainer = shap.TreeExplainer(model_GBT)
    return model_GBT, base_value, explainer







def shap_dependence_plot(X_train, explainer):
    """
    Computes necessary steps to generate shap dependence plot and shap feauture importance

    Arguments:
    - X_train: train set for the features 
    - explainer: SHAP tree explainer

    Returns:
    - feature_importance_name: list containing the name of the features order by importance
    - feature_importance_value: numpy array containing the importance value per feature
    - temp_df: dataframe containing X_train and shapeley values for each instance/feature
    - liste_shap_features: list containing the name of shap features

    """

    shap_values = explainer.shap_values(X_train)
    dataframe_shap = pd.DataFrame(
        shap_values,
        columns=list(map(lambda x: x + "_shap", X_train.columns.tolist())),
    )
    dataframe_shap = dataframe_shap[
        dataframe_shap.abs().sum().sort_values(ascending=False).index.tolist()
    ]
    feature_importance = (
        dataframe_shap.abs().sum().sort_values(ascending=False)
    )
    feature_importance_name = feature_importance.index.tolist()
    feature_importance_value = feature_importance.values
    dataframe_shap.index = X_train.index
    temp_df = pd.concat([X_train, dataframe_shap], 1)
    liste_shap_features = list(
        filter(lambda x: x.endswith("_shap"), temp_df.columns.tolist())
    )
    return (
        feature_importance_name,
        feature_importance_value,
        temp_df,
        liste_shap_features,
    )

def shap_single_explanation(explainer, X_test, explanation, model_GBT, base_value):
    """
    Compute a single force plot for an instance from the test set

    Arguments:
    - explainer: SHAP tree explainer
    - X_test: test set for the features 
    - explanation: index of X_test (instance to explain)
    - model_GBT: trained gradient boosting regressor model
    - base_value: mean value of the predictions on the train set

    Returns:
    - feature_importance_single_explanation_value: sorted importance values for the features
    - sum_list: list containing  strings of the name of each feature and its value 
    - color: Blue for negative values and Crimson for the positive values
    - title_single: title of shap force plot

    """
    dataframe_single_explanation = pd.DataFrame(
        [explainer.shap_values(X_test.iloc[explanation, :])],
        columns=X_test.columns,
    )
    sorted_importance = dataframe_single_explanation.iloc[0, :].sort_values(
        ascending=False
    )
    feature_importance_single_explanation_name = (
        sorted_importance.index.tolist()
    )
    feature_importance_single_explanation_value = sorted_importance.values
    color = np.array(
        ["rgb(255,255,255)"]
        * feature_importance_single_explanation_value.shape[0]
    )
    color[feature_importance_single_explanation_value < 0] = "Blue"
    color[feature_importance_single_explanation_value > 0] = "Crimson"
    list_ordered_values = X_test.iloc[explanation, :][
        feature_importance_single_explanation_name
    ].values
    sum_list = []
    for (item1, item2) in zip(
        feature_importance_single_explanation_name, list_ordered_values
    ):
        sum_list.append(" = ".join([item1, str(item2)]))
    predicted_value = str(
        round(
            model_GBT.predict(
                np.array(X_test.iloc[explanation, :]).reshape(1, -1)
            )[0],
            2,
        )
    )
    title_single = "Feature importance: Base value: {} , Predicted value: {}".format(
        base_value, predicted_value
    )
    return (
        feature_importance_single_explanation_value,
        sum_list,
        color,
        title_single,
    )

def create_explanations(X_train, X_test, y_train):
    """
    Create SHAP explanations : feature importance, dependence plot and force plot.

    Arguments:
    - X_train: train set for the features
    - X_test: test set for the features
    - y_train: train set for the target variable

    Returns:
    - model_GBT: trained gradient boosting regressor model
    - base_value: mean value of the predictions on the train set
    - explainer: SHAP tree explainer
    - feature_importance_name: list containing the name of the features order by importance
    - feature_importance_value: numpy array containing the importance value per feature
    - temp_df: dataframe containing X_train and shapeley values for each instance/feature
    - feature_importance_single_explanation_value:
    - sum_list: list containing  strings of the name of each feature and its value 
    - title_single: title of shap force plot
    - color: Blue for negative values and Crimson for the positive values
    - liste_shap_features: list containing the name of shap features
    """

    model_GBT, base_value, explainer = create_explainer(X_train, y_train)
    (
        feature_importance_name,
        feature_importance_value,
        temp_df,
        liste_shap_features,
    ) = shap_dependence_plot(X_train, explainer)
    (
        feature_importance_single_explanation_value,
        sum_list,
        color,
        title_single,
    ) = shap_single_explanation(explainer, X_test, 0, model_GBT, base_value)

    return (
        model_GBT,
        base_value,
        explainer,
        feature_importance_name,
        feature_importance_value,
        temp_df,
        feature_importance_single_explanation_value,
        sum_list,
        title_single,
        color,
        liste_shap_features,
    )
