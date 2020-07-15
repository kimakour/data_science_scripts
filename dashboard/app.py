import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from numpy import mean
from numpy import absolute
from scipy.integrate import simps
import shap
from sklearn.ensemble import GradientBoostingRegressor

app = dash.Dash(__name__)

################################################# Functions #########
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


########################################## Dataframes and Models #####################################################################################
df = pd.read_csv("datasets_229906_491820_Fish.csv")
encoded_df = pd.get_dummies(df, drop_first=True)
encoded_df = encoded_df[
    list(set(encoded_df.columns.tolist()) - set(["Width"])) + ["Width"]
]

X, y = encoded_df.iloc[:, :-1], encoded_df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    encoded_df.iloc[:, :-1], encoded_df.iloc[:, -1]
)

pipeline = Pipeline(
    steps=[("normalize", MinMaxScaler()), ("model", LinearRegression())]
)

cv = KFold(n_splits=5, shuffle=True, random_state=1)
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
y_test = pd.DataFrame(y_test).rename(columns={"Width": "y_test"})
y_predicted = pd.DataFrame(y_pred, columns={"y_pred"})
y_predicted.index = y_test.index
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

model_GBT = GradientBoostingRegressor().fit(X_train, y_train)
explainer = shap.TreeExplainer(model_GBT)
shap_values = explainer.shap_values(X_train)
dataframe_shap = pd.DataFrame(
    shap_values,
    columns=list(map(lambda x: x + "_shap", encoded_df.columns[:-1].tolist())),
)
dataframe_shap = dataframe_shap[
    dataframe_shap.abs().sum().sort_values(ascending=False).index.tolist()
]
feature_importance = dataframe_shap.abs().sum().sort_values(ascending=False)
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
feature_importance_single_explanation_name = sorted_importance.index.tolist()
feature_importance_single_explanation_value = sorted_importance.values

color = np.array(
    ["rgb(255,255,255)"] * feature_importance_single_explanation_value.shape[0]
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
    round(model_GBT.predict(np.array(X_test.iloc[0, :]).reshape(1, -1))[0], 2)
)
title_single = "Feature importance: Base value: {} , Predicted value: {}".format(
    base_value, predicted_value
)


# liste_indexes = dataframe_error[(df_predictions['True']<20) & (df_predictions['predicted']>100)].index
########################################## Plots of the beginning #######################################################################################
style_plot = {
    "border-radius": "5px",
    "background-color": "#f9f9f9",
    "box-shadow": "2px 2px 2px lightgrey",
    "margin": "10px",
}
fig1 = px.scatter(
    df,
    x=df.columns[0],
    y=df.columns[0],
    marginal_x="histogram",
    marginal_y="histogram",
    title=" Scatter plot",
)
fig1.update_layout(
    paper_bgcolor="#f9f9f9",
    title={"y": 0.9, "x": 0.5, "xanchor": "center", "yanchor": "top"},
)
fig2 = px.histogram(df, x=df.columns[0], title="Histogram plot")
fig2.update_layout(
    paper_bgcolor="#f9f9f9",
    title={"y": 0.9, "x": 0.5, "xanchor": "center", "yanchor": "top"},
)
fig3 = px.density_heatmap(
    df,
    x=df.columns[0],
    y=df.columns[0],
    marginal_x="histogram",
    marginal_y="histogram",
    title="Density Heatmap plot",
)
fig3.update_layout(
    paper_bgcolor="#f9f9f9",
    title={"y": 0.9, "x": 0.5, "xanchor": "center", "yanchor": "top"},
)
fig4 = go.Figure(
    data=[go.Pie(labels=labels, values=values, sort=False)],
    layout={"title": "Pie chart for the absolute error"},
)
fig4.update_layout(
    paper_bgcolor="#f9f9f9",
    title={"y": 0.9, "x": 0.5, "xanchor": "center", "yanchor": "top"},
)
fig5 = px.scatter(
    dataframe_error,
    x="y_pred",
    y="error",
    title=" Scatter plot for the error and the prediction",
)
fig5.update_layout(
    paper_bgcolor="#f9f9f9",
    title={"y": 0.9, "x": 0.5, "xanchor": "center", "yanchor": "top"},
)

fig6 = px.line(df_REC, x="Deviance", y="Accuracy", title=title_REC)
fig6.update_layout(
    paper_bgcolor="#f9f9f9",
    title={"y": 0.9, "x": 0.5, "xanchor": "center", "yanchor": "top"},
)

fig7 = go.Figure(
    [go.Bar(x=feature_importance_name, y=feature_importance_value)],
    layout={"title": "Feature importance"},
)
fig7.update_layout(
    paper_bgcolor="#f9f9f9",
    title={"y": 0.9, "x": 0.5, "xanchor": "center", "yanchor": "top"},
)

fig8 = px.scatter(
    temp_df,
    x="Weight",
    y="Weight_shap",
    color="Weight",
    title="SHAP dependence plot",
)
fig8.update_layout(
    paper_bgcolor="#f9f9f9",
    title={"y": 0.9, "x": 0.5, "xanchor": "center", "yanchor": "top"},
)
fig9 = go.Figure(
    [
        go.Bar(
            x=feature_importance_single_explanation_value,
            y=sum_list,
            orientation="h",
            marker_color=color,
        )
    ],
    layout={"title": title_single},
)
fig9.update_layout(
    paper_bgcolor="#f9f9f9",
    title={"y": 0.9, "x": 0.5, "xanchor": "center", "yanchor": "top"},
)
app.layout = html.Div(
    style={"backgroundColor": "#F9F9F9"},
    children=[
        ##################################################################################################
        ##################################### Header #####################################################
        html.Div(
            [
                html.Div(
                    [
                        html.Img(
                            src=app.get_asset_url("magicarpe.png"),
                            id="plotly-image",
                            style={
                                "height": "60px",
                                "width": "auto",
                                "margin-left": "50px",
                                "margin-top": "10px",
                                "margin-bottom": "25px",
                            },
                        )
                    ],
                    className="one-third column",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    "Data Visualization for fish Data set",
                                    style={"margin-bottom": "0px"},
                                ),
                            ]
                        )
                    ],
                    className="one-third column",
                    id="title",
                ),
                html.Div(
                    [
                        html.Img(
                            src=app.get_asset_url("plotly.png"),
                            id="plotly-image-2",
                            style={
                                "height": "60px",
                                "width": "auto",
                                "margin-left": "400px",
                                "margin-top": "10px",
                                "margin-bottom": "25px",
                            },
                        )
                    ],
                    className="one-third column",
                ),
            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"},
        ),
        html.Div(
            [
                html.H4(
                    "Basic visualizations for the features",
                    style={"textAlign": "center"},
                ),
                html.Div(
                    [
                        ##################################################################################################
                        ########################### Histogram Plot ###################################################
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.P(
                                                    "Choose the feature you want to plot for the histogram"
                                                )
                                            ],
                                            className="row",
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        create_dcc(
                                                            df.columns,
                                                            "value_histo",
                                                            df.columns[0],
                                                        )
                                                    ],
                                                    className="one-half column",
                                                )
                                            ],
                                            className="row",
                                        ),
                                        dcc.Graph(
                                            id="histogram",
                                            figure=fig2,
                                            style=style_plot,
                                        ),
                                    ]
                                )
                            ],
                            className="one-third column pretty_container",
                        ),
                        ##################################################################################################
                        ########################### Scatter plot ##########################################
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.P(
                                            "Choose the features you want to plot for the scatter plot"
                                        )
                                    ],
                                    className="row",
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        create_dcc(
                                                            df.columns,
                                                            "value_x_scatter",
                                                            df.columns[0],
                                                        )
                                                    ],
                                                    className="row",
                                                ),
                                                html.Div(
                                                    [
                                                        create_dcc(
                                                            df.columns,
                                                            "value_y_scatter",
                                                            df.columns[0],
                                                        )
                                                    ],
                                                    className="row",
                                                ),
                                                html.Div(
                                                    [
                                                        create_slider(
                                                            "value_slider2",
                                                            0,
                                                            1,
                                                            {
                                                                0: "None",
                                                                1: "Output",
                                                            },
                                                        )
                                                    ],
                                                    className="row",
                                                ),
                                            ],
                                            className="one-half column",
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        create_radio_shape(
                                                            "radio_value_x_scatter"
                                                        )
                                                    ],
                                                    className="one-half column",
                                                ),
                                                html.Div(
                                                    [
                                                        create_radio_shape(
                                                            "radio_value_y_scatter"
                                                        )
                                                    ],
                                                    className="one-half column",
                                                ),
                                            ],
                                            className="one-half column",
                                        ),
                                    ],
                                    className="row",
                                ),
                                dcc.Graph(
                                    id="scatter-plot",
                                    figure=fig1,
                                    style=style_plot,
                                ),
                            ],
                            className="one-third column pretty_container",
                        ),
                        ##################################################################################################
                        ############################ Histogram 2D ###########################################
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.P(
                                            "Choose the features you want to plot the 2D histogram"
                                        )
                                    ],
                                    className="row",
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                create_dcc(
                                                    df.columns,
                                                    "value_x_histo",
                                                    df.columns[0],
                                                )
                                            ],
                                            className="one-half column",
                                        ),
                                        html.Div(
                                            [
                                                create_dcc(
                                                    df.columns,
                                                    "value_y_histo",
                                                    df.columns[0],
                                                )
                                            ],
                                            className="one-half column",
                                        ),
                                    ],
                                    className="row",
                                ),
                                dcc.Graph(
                                    id="2D_histo",
                                    figure=fig3,
                                    style=style_plot,
                                ),
                            ],
                            className="one-third column pretty_container",
                        ),
                    ],
                    className="row",
                ),
            ],
            className="row pretty_container",
        ),
        html.Div(
            [
                html.H4(" Model performances", style={"textAlign": "center"}),
                html.Div(
                    [
                        ################################################################################################################
                        ################################### model metrics & pie chart############################################################
                        html.Div(
                            [
                                ####### Table for metrics
                                html.Div(
                                    [
                                        dash_table.DataTable(
                                            id="table",
                                            columns=[
                                                {"name": i, "id": i}
                                                for i in dataframe_results.columns
                                            ],
                                            data=dataframe_results.to_dict(
                                                "records"
                                            ),
                                        )
                                    ],
                                    className="row",
                                ),
                                ####### Pie chart
                                html.Div(
                                    [
                                        dcc.Graph(
                                            id="pie-chart",
                                            figure=fig4,
                                            style=style_plot,
                                        )
                                    ],
                                    className="row",
                                ),
                            ],
                            className="one-third column pretty_container",
                        ),
                        ################################################## scatter plots for the error ##############################
                        html.Div(
                            [
                                html.Div(
                                    [
                                        dcc.Dropdown(
                                            id="error_id",
                                            options=[
                                                {"label": c, "value": c}
                                                for c in dataframe_error.columns[
                                                    :-1
                                                ]
                                            ],
                                            value="error",
                                        )
                                    ],
                                    className="row",
                                ),
                                dcc.Graph(
                                    id="error-scatter",
                                    figure=fig5,
                                    style=style_plot,
                                ),
                            ],
                            className="one-third column pretty_container",
                        ),
                        html.Div(
                            [
                                dcc.Graph(
                                    id="REC-plot",
                                    figure=fig6,
                                    style=style_plot,
                                )
                            ],
                            className="one-third column pretty_container",
                        ),
                    ],
                    className="row",
                ),
            ],
            className="row pretty_container",
        ),
        html.Div(
            [
                html.H4(
                    "Model Interpretability using SHAP values",
                    style={"textAlign": "center"},
                ),
                html.Div(
                    [
                        dcc.Graph(
                            id="feature-importance-plot",
                            figure=fig7,
                            style=style_plot,
                        )
                    ],
                    className="one-third column pretty_container",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        create_dcc(
                                            liste_shap_features,
                                            "id-shap-feature",
                                            liste_shap_features[0],
                                        )
                                    ],
                                    className="one-third column",
                                ),
                                html.Div(
                                    [
                                        create_dcc(
                                            encoded_df.columns,
                                            "id-feature1",
                                            encoded_df.columns[0],
                                        )
                                    ],
                                    className="one-third column",
                                ),
                                html.Div(
                                    [
                                        create_dcc(
                                            encoded_df.columns,
                                            "id-feature2",
                                            encoded_df.columns[0],
                                        )
                                    ],
                                    className="one-third column",
                                ),
                            ],
                            className="row pretty_container",
                        ),
                        html.Div(
                            [
                                dcc.Graph(
                                    id="dependence-plot",
                                    figure=fig8,
                                    style=style_plot,
                                )
                            ],
                            className="row",
                        ),
                    ],
                    className="one-third column pretty_container",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.P(
                                            "Choose a row from the Test set to explain"
                                        )
                                    ],
                                    className="two-thids column",
                                ),
                                html.Div(
                                    [
                                        dcc.Dropdown(
                                            id="explanation",
                                            options=[
                                                {"label": i, "value": i}
                                                for i in range(len(X_test))
                                            ],
                                            value=0,
                                        )
                                    ],
                                    className="one-third column",
                                ),
                            ],
                            className="row",
                        ),
                        dcc.Graph(
                            id="single-explanation-plot",
                            figure=fig9,
                            style=style_plot,
                        ),
                    ],
                    className="one-third column pretty_container",
                ),
            ],
            className="row pretty_container",
        ),
    ],
)


@app.callback(Output("error-scatter", "figure"), [Input("error_id", "value")])
def plot_scatter_error(y_data):
    figure = px.scatter(
        dataframe_error,
        x="y_pred",
        y=y_data,
        title="Scatter for the prediction",
    )
    figure.update_layout(
        paper_bgcolor="#f9f9f9",
        title={"y": 0.9, "x": 0.5, "xanchor": "center", "yanchor": "top"},
    )
    return figure


@app.callback(
    Output("scatter-plot", "figure"),
    [
        Input("value_x_scatter", "value"),
        Input("value_y_scatter", "value"),
        Input("radio_value_x_scatter", "value"),
        Input("radio_value_y_scatter", "value"),
        Input("value_slider2", "value"),
    ],
)
def plot_scatter(x_data, y_data, x_radio, y_radio, slider2):
    if slider2 == 0:
        figure = px.scatter(
            df,
            x=x_data,
            y=y_data,
            marginal_x=x_radio,
            marginal_y=y_radio,
            log_x=False,
            title=" Scatter plot",
        )
    else:
        figure = px.scatter(
            df,
            x=x_data,
            y=y_data,
            marginal_x=x_radio,
            marginal_y=y_radio,
            color="OUTPUT",
            log_x=False,
            title=" Scatter plot",
        )
    figure.update_layout(
        paper_bgcolor="#f9f9f9",
        title={"y": 0.9, "x": 0.5, "xanchor": "center", "yanchor": "top"},
    )
    return figure


@app.callback(Output("histogram", "figure"), [Input("value_histo", "value")])
def plot_histogram(x_data):
    figure = px.histogram(df, x=x_data, title="Histogram plot")
    figure.update_layout(
        paper_bgcolor="#f9f9f9",
        title={"y": 0.9, "x": 0.5, "xanchor": "center", "yanchor": "top"},
    )
    return figure


@app.callback(
    Output("2D_histo", "figure"),
    [Input("value_x_histo", "value"), Input("value_y_histo", "value")],
)
def plot_2D_histogram(value_x_histo, value_y_histo):
    figure = px.density_heatmap(
        df,
        x=value_x_histo,
        y=value_y_histo,
        marginal_x="histogram",
        marginal_y="histogram",
        title="Density Heatmap plot",
    )
    figure.update_layout(
        paper_bgcolor="#f9f9f9",
        title={"y": 0.9, "x": 0.5, "xanchor": "center", "yanchor": "top"},
    )
    return figure


@app.callback(
    Output("dependence-plot", "figure"),
    [
        Input("id-shap-feature", "value"),
        Input("id-feature1", "value"),
        Input("id-feature2", "value"),
    ],
)
def plot_dependence_shap(id_shap_feature, id_feature1, id_feature2):
    figure = px.scatter(
        temp_df,
        x=id_feature1,
        y=id_shap_feature,
        color=id_feature2,
        title="SHAP dependence plot",
    )
    figure.update_layout(
        paper_bgcolor="#f9f9f9",
        title={"y": 0.9, "x": 0.5, "xanchor": "center", "yanchor": "top"},
    )
    return figure


@app.callback(
    Output("single-explanation-plot", "figure"),
    [Input("explanation", "value")],
)
def plot_single_explanation(explanation):
    dataframe_single_explanation = pd.DataFrame(
        [explainer.shap_values(X_test.iloc[explanation, :])],
        columns=X_train.columns,
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
        sum_list.append(item1 + " = " + str(item2))
    # base_value = str(round(model_GBT.predict(X_train).mean(),2))
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
    figure = go.Figure(
        [
            go.Bar(
                x=feature_importance_single_explanation_value,
                y=sum_list,
                orientation="h",
                marker_color=color,
            )
        ],
        layout={"title": title_single},
    )
    figure.update_layout(
        paper_bgcolor="#f9f9f9",
        title={"y": 0.9, "x": 0.5, "xanchor": "center", "yanchor": "top"},
    )
    return figure


if __name__ == "__main__":
    app.run_server(debug=True)
