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


def REC(y_true, y_pred, min_Range, max_Range, Interval_Size):
    Accuracy = []
    Epsilon = np.arange(min_Range, max_Range, Interval_Size)
    for i in range(len(Epsilon)):
        count = 0.0
        for j in range(len(y_true)):
            if np.abs(y_true[j] - y_pred[j]) < Epsilon[i]:
                count = count + 1
        Accuracy.append(count / len(y_true))
    AUC = simps(Accuracy, Epsilon) / max_Range
    return Epsilon, Accuracy, AUC


def intervals(x):
    if x < 0.1:
        return "absolute error < 0.1 "
    elif x >= 0.1 and x < 0.5:
        return "0.1 < absolute error < 0.5 "
    elif x >= 0.5 and x < 1:
        return "0.5 < absolute error < 1 "
    else:
        return "absolute error > 1 "


def create_dcc(possible_values, name, default_value):
    return dcc.Dropdown(
        id=name,
        options=[{"label": c, "value": c} for c in possible_values],
        value=default_value,
    )


def create_slider(name, mini, maxi, marks):
    return dcc.Slider(
        id=name,
        min=mini,
        max=maxi,
        marks=marks,
        value=0,
        className="pretty_container",
    )


def create_radio_shape(name):
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
    encoded_df = pd.get_dummies(df, drop_first=is_linear_regression)
    encoded_df = encoded_df[
        list(set(encoded_df.columns.tolist()) - set(["Width"])) + ["Width"]
    ]
    return encoded_df


def create_dataframe_results(X, y, X_train, X_test, y_train, y_test, kfolds=5):

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
        columns=[" mean crossval MSE", "R2 score", "MSE", "MAE"],
    )

    dataframe_absolute_error = pd.DataFrame(np.abs(y_test - y_pred))
    dataframe_absolute_error["intervals"] = dataframe_absolute_error.apply(
        lambda x: intervals(x["Width"]), axis=1
    )

    return dataframe_results, dataframe_absolute_error, y_pred


def create_values_pie_chart(dataframe_absolute_error):
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
    error = pd.DataFrame(
        y_test["y_test"] - y_predicted["y_pred"], columns=["error"]
    )
    error.index = y_test.index
    dataframe_error = pd.concat([error, y_test, y_predicted], 1)
    deviance, accuracy, AUC = REC(
        dataframe_error["y_test"].values,
        dataframe_error["y_pred"].values,
        0,
        3,
        0.05,
    )
    d = {"Deviance": deviance, "Accuracy": accuracy}
    df_REC = pd.DataFrame(data=d)
    title_REC = "REC curve , AUC score=" + str(round(AUC, 3))
    return dataframe_error, df_REC, title_REC


def create_explainer(X_train, X_test, y_train, encoded_df):
    model_GBT = GradientBoostingRegressor().fit(X_train, y_train)
    explainer = shap.TreeExplainer(model_GBT)
    shap_values = explainer.shap_values(X_train)
    dataframe_shap = pd.DataFrame(
        shap_values,
        columns=list(
            map(lambda x: x + "_shap", encoded_df.columns[:-1].tolist())
        ),
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
    dataframe_single_explanation = pd.DataFrame(
        [explainer.shap_values(X_test.iloc[0, :])], columns=X_train.columns
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
    list_ordered_values = X_test.iloc[0, :][
        feature_importance_single_explanation_name
    ].values
    sum_list = []
    for (item1, item2) in zip(
        feature_importance_single_explanation_name, list_ordered_values
    ):
        sum_list.append(item1 + " = " + str(item2))
    base_value = str(round(model_GBT.predict(X_train).mean(), 2))
    predicted_value = str(
        round(
            model_GBT.predict(np.array(X_test.iloc[0, :]).reshape(1, -1))[0], 2
        )
    )
    title_single = "Feature importance: Base value: {} , Predicted value: {}".format(
        base_value, predicted_value
    )
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
