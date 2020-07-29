import plotly.express as px
import plotly.graph_objects as go


def fig_update_layout(fig):
    fig.update_layout(
        paper_bgcolor="#f9f9f9",
        title={"y": 0.9, "x": 0.5, "xanchor": "center", "yanchor": "top"},
    )
    return fig


def fig_1(df):
    fig = px.scatter(
        df,
        x=df.columns[0],
        y=df.columns[0],
        marginal_x="histogram",
        marginal_y="histogram",
        title=" Scatter plot",
    )
    fig = fig_update_layout(fig)
    return fig


def fig_2(df):
    fig = px.histogram(df, x=df.columns[0], title="Histogram plot")
    fig = fig_update_layout(fig)
    return fig


def fig_3(df):
    fig = px.density_heatmap(
        df,
        x=df.columns[0],
        y=df.columns[0],
        marginal_x="histogram",
        marginal_y="histogram",
        title="Density Heatmap plot",
    )
    fig = fig_update_layout(fig)
    return fig


def fig_4(labels, values_pie):
    fig = go.Figure(
        data=[go.Pie(labels=labels, values=values_pie, sort=False)],
        layout={"title": "Pie chart for the absolute error"},
    )
    fig = fig_update_layout(fig)
    return fig


def fig_5(dataframe_error):
    fig = px.scatter(
        dataframe_error,
        x="y_pred",
        y="error",
        title=" Scatter plot for the error and the prediction",
    )
    fig = fig_update_layout(fig)
    return fig


def fig_6(df_REC, title_REC):
    fig = px.line(df_REC, x="Deviance", y="Accuracy", title=title_REC)
    fig = fig_update_layout(fig)
    return fig


def fig_7(feature_importance_name, feature_importance_value):
    fig = go.Figure(
        [go.Bar(x=feature_importance_name, y=feature_importance_value)],
        layout={"title": "Feature importance"},
    )
    fig = fig_update_layout(fig)
    return fig


def fig_8(temp_df):
    fig = px.scatter(
        temp_df,
        x="Weight",
        y="Weight_shap",
        color="Weight",
        title="SHAP dependence plot",
    )
    fig = fig_update_layout(fig)
    return fig


def fig_9(feature_importance_single_explanation_value, sum_list, color, title_single):
    fig = go.Figure(
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
    fig = fig_update_layout(fig)
    return fig
