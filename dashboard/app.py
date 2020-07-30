import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
from sklearn.model_selection import train_test_split

from func_dash.func_dash import (
    create_dcc,
    create_slider,
    create_radio_shape,
    re_order_dataset,
    create_dataframe_results,
    create_values_pie_chart,
    create_REC_plot,
    shap_single_explanation,
    create_explanations,
)
from func_dash.init_fig_dash import (
    fig_scatter,
    fig_hist,
    fig_density_heatmap,
    fig_pie,
    fig_REC,
    fig_bar_shap,
    fig_force_plot,
)

# initialize the app
app = dash.Dash(__name__)


# load the dataframe
df = pd.read_csv("datasets_229906_491820_Fish.csv")
encoded_df = re_order_dataset(df)
X, y = encoded_df.iloc[:, :-1], encoded_df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    encoded_df.iloc[:, :-1], encoded_df.iloc[:, -1]
)
# calculate necessary computations to run the data analysis, model performance and shap values
dataframe_results, dataframe_absolute_error, y_pred = create_dataframe_results(
    X, y, X_train, X_test, y_train, y_test
)

labels, values_pie = create_values_pie_chart(dataframe_absolute_error)

y_test = pd.DataFrame(y_test).rename(columns={"Width": "y_test"})
y_predicted = pd.DataFrame(y_pred, columns={"y_pred"})
y_predicted.index = y_test.index

dataframe_error, df_REC, title_REC = create_REC_plot(y_test, y_predicted)
(
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
) = create_explanations(X_train, X_test, y_train)


# Plots at the initialization
style_plot = {
    "border-radius": "5px",
    "background-color": "#f9f9f9",
    "box-shadow": "2px 2px 2px lightgrey",
    "margin": "10px",
}
fig1 = fig_hist(df, df.columns[0], "Histogram plot for the features")

fig2 = fig_scatter(
    df,
    df.columns[0],
    df.columns[0],
    "Scatter plot for features",
    marginal_x="histogram",
    marginal_y="histogram",
)

fig3 = fig_density_heatmap(
    df,
    df.columns[0],
    df.columns[0],
    "Density Heatmap plot",
    marginal_x="histogram",
    marginal_y="histogram",
)
fig4 = fig_pie(labels, values_pie, "Pie chart for the absolute error")
fig5 = fig_scatter(
    dataframe_error,
    "y_pred",
    "error",
    "Scatter plot for the error and the prediction",
)
fig6 = fig_REC(df_REC, title_REC)
fig7 = fig_bar_shap(feature_importance_name, feature_importance_value)
fig8 = fig_scatter(
    temp_df, "Weight", "Weight_shap", "SHAP dependence plot", color="Weight"
)
fig9 = fig_force_plot(
    feature_importance_single_explanation_value, sum_list, color, title_single
)

# application layout

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
                                            figure=fig1,
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
                                    figure=fig2,
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
@app.callback(Output("histogram", "figure"), [Input("value_histo", "value")])
def plot_histogram(x_data):
    figure = fig_hist(df, x_data, "Histogram plot")
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
        figure = fig_scatter(
            df,
            x_data,
            y_data,
            "Scatter plot",
            marginal_x=x_radio,
            marginal_y=y_radio,
        )
    else:
        figure = fig_scatter(
            df,
            x_data,
            y_data,
            "Scatter plot",
            marginal_x=x_radio,
            marginal_y=y_radio,
            color="Width",
        )
    return figure

@app.callback(
    Output("2D_histo", "figure"),
    [Input("value_x_histo", "value"), Input("value_y_histo", "value")],
)
def plot_2D_histogram(value_x_histo, value_y_histo):
    figure = fig_density_heatmap(
        df,
        value_x_histo,
        value_y_histo,
        "Density Heatmap plot",
        marginal_x="histogram",
        marginal_y="histogram",
    )
    return figure

@app.callback(Output("error-scatter", "figure"), [Input("error_id", "value")])
def plot_scatter_error(y_data):
    figure = fig_scatter(
        dataframe_error, "y_pred", y_data, "Scatter for the prediction"
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
    figure = fig_scatter(
        temp_df,
        id_feature1,
        id_shap_feature,
        "SHAP dependence plot",
        color=id_feature2,
    )
    return figure


@app.callback(
    Output("single-explanation-plot", "figure"),
    [Input("explanation", "value")],
)
def plot_single_explanation(explanation):
    (
        feature_importance_single_explanation_value,
        sum_list,
        color,
        title_single,
    ) = shap_single_explanation(
        explainer, X_test, explanation, model_GBT, base_value
    )
    figure = fig_force_plot(
        feature_importance_single_explanation_value,
        sum_list,
        color,
        title_single,
    )
    return figure


if __name__ == "__main__":
    app.run_server(debug=True)
