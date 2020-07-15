import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from model_evaluation_functions import read_dataset

# from DS_functions_model_interpretability.PostPredictionAnalysis import compute_adjusted_cosine_similarity

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


app = dash.Dash(__name__)

################################################# Functions #########
def compute_adjusted_cosine_similarity(
    feature_importances, X_train, X_test, y_test, list_indexes
):
    """
    Computing an adjusted cosine similarity between two dataframes
    feature_importances : the feature importance score for each feature 
    X_train : pandas dataframe/series 
    X_test : pandas dataframe/series
    list_indexes : list of indexes for the X_test that we want to know their similarity with instances in X_train
    returns:
    - pandas dataframe containing the cosine similarity for each vector between X_train and X_test
    """
    mmsc = MinMaxScaler().fit(X_train)
    a = np.multiply(feature_importances, mmsc.transform(X_train))
    b = np.multiply(feature_importances, mmsc.transform(X_test))
    return pd.DataFrame(
        cosine_similarity(a, b),
        columns=y_test.loc[list_indexes, "y_test"].to_list(),
    )


def create_dcc(name):
    return dcc.Dropdown(
        id=name,
        options=[{"label": c, "value": c} for c in df.columns],
        value="Activity.Duration",
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
df = pd.read_csv('Accidental_Drug_Related_Deaths_2012-2018.csv')
liste_columns = df.columns.tolist()

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
    x=liste_columns[3],
    y=liste_columns[3],
    marginal_x="histogram",
    marginal_y="histogram",
    title=" Scatter plot",
)
fig1.update_layout(
    title={"y": 0.9, "x": 0.5, "xanchor": "center", "yanchor": "top"}
)
fig2 = px.histogram(df, x=liste_columns[3], title="Histogram plot")
fig2.update_layout(
    title={"y": 0.9, "x": 0.5, "xanchor": "center", "yanchor": "top"}
)
fig3 = px.density_heatmap(
    df,
    x=liste_columns[3],
    y=liste_columns[3],
    marginal_x="histogram",
    marginal_y="histogram",
    title="Density Heatmap plot",
)
fig3.update_layout(
    title={"y": 0.9, "x": 0.5, "xanchor": "center", "yanchor": "top"}
)
fig4 = go.Figure(
    data=[go.Pie(labels=labels, values=values, sort=False)],
    layout={"title": "Pie chart for the absolute error"},
)
fig4.update_layout(
    title={"y": 0.9, "x": 0.5, "xanchor": "center", "yanchor": "top"}
)
fig5 = px.scatter(
    dataframe_error,
    x="y_pred",
    y="error",
    title=" Scatter plot for the error and the prediction",
)
fig5.update_layout(
    title={"y": 0.9, "x": 0.5, "xanchor": "center", "yanchor": "top"}
)
layout = go.Layout(xaxis=dict(type="category"))
fig6 = px.box(
    generic_overpredicted_dataframe,
    x="instance",
    y="cos",
    color="value",
    title="Adjusted Cosine Similarity",
)
fig6.update_layout(xaxis=dict(type="category"))
fig6.update_layout(
    title={"y": 0.9, "x": 0.5, "xanchor": "center", "yanchor": "top"}
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
                            src=app.get_asset_url("plw.png"),
                            id="plotly-image",
                            style={
                                "height": "60px",
                                "width": "auto",
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
                                    "Data Visualization for UCB Data set",
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
                            src=app.get_asset_url("ucb.png"),
                            id="plotly-image-2",
                            style={
                                "height": "60px",
                                "width": "auto",
                                "margin-left": "450px",
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
                                                            "value_histo"
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
                                                            "value_x_scatter"
                                                        )
                                                    ],
                                                    className="row",
                                                ),
                                                html.Div(
                                                    [
                                                        create_dcc(
                                                            "value_y_scatter"
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
                                            [create_dcc("value_x_histo")],
                                            className="one-half column",
                                        ),
                                        html.Div(
                                            [create_dcc("value_y_histo")],
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
       
    ],
)




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
        title={"y": 0.9, "x": 0.5, "xanchor": "center", "yanchor": "top"}
    )
    return figure


@app.callback(Output("histogram", "figure"), [Input("value_histo", "value")])
def plot_histogram(x_data):
    figure = px.histogram(df, x=x_data, title="Histogram plot")
    figure.update_layout(
        title={"y": 0.9, "x": 0.5, "xanchor": "center", "yanchor": "top"}
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
        title={"y": 0.9, "x": 0.5, "xanchor": "center", "yanchor": "top"}
    )
    return figure


if __name__ == "__main__":
    app.run_server(debug=True)
