import itertools

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
import numpy as np
import scipy.stats as ss


def density_dataframe(
    dataframe,
    number_columns,
    number_rows,
    figsize=(15, 15),
    hist_true_false=False,
):
    """
	plot the densities for all the features in a dataframe 
	dataframe: pandas dataframe
	number_columns: number of columns for the subplot 
	number_rows: number of rows for the subplot
	figsize : Figure size, default=(15,15)
	hist_true_false : Boolean for depicting histograms with densities or not, default= False
	"""
    plt.figure(
        num=None, figsize=figsize, dpi=100, facecolor="w", edgecolor="k"
    )
    for i, name in enumerate(dataframe.columns[:-1]):
        plt.subplot(number_rows, number_columns, i + 1)
        sns.distplot(dataframe[name], hist=hist_true_false, label=name)
    plt.title("Density for the features")


# showing the proportion of a categorical feature by the values of another feature
def proportion(
    df,
    cat_feature,
    feature,
    liste=[0, 10, 20, 50, 100, 200, 450],
    x_subplot=3,
    y_subplot=3,
):
    plt.figure(
        num=None, figsize=(15, 15), dpi=100, facecolor="w", edgecolor="k"
    )
    xaxes = []
    j = 1
    plt.title("Histogram of the class when the number of the feature exceeds")
    for i in liste:
        xaxes.append(i)
        plt.subplot(x_subplot, y_subplot, j)
        plt.title(str(i))
        plt.hist(df[str(cat_feature)][df[str(feature)] > i])
        j += 1


# expanding mean
def density_variables_expanding_mean(dataframe, x_axis=4, y_axis=3):
    """
	plot the expanding mean over time for the features of an ordered dataframe
	dataframe: pandas dataframe
	x_axis: number of columns for the subplot
	y_axis: number of rows for the subplot
	"""
    plt.figure(
        num=None, figsize=(15, 15), dpi=100, facecolor="w", edgecolor="k"
    )
    for i, name in enumerate(dataframe.columns[:-1]):
        plt.subplot(x_axis, y_axis, i + 1)
        plt.plot(dataframe[name].expanding(axis=0).mean())
        plt.title(name, y=-0.2)


def boxplot(df, feature, feature1, showfliers=False):
    """
	Boxplot for a feature according to another one
	df: dataframe
	feature: feature on the x axis
	feature1: feature on the y axis
	showfliers: whether to show values out of the IQR method interval , default = False
	"""
    ax = sns.boxplot(
        x=feature, y=feature1, data=df, palette="Set3", showfliers=showfliers
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=70)


def boxplot_maxmin(df, feature, feature1, y_min, y_max):
    """
	Boxplot one variable according to another variable with a threshold of the values 
	df: pandas dataframe
	feature: feature on the x axis
	feature1: feature on the y axis
	y_min: the bottom limit
	y_max: the top limit
	"""
    ax = sns.boxplot(x=feature, y=feature1, data=df, palette="Set3")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=70)
    ax.axis(ymin=y_min, ymax=y_max)


# Boxplot of all the variables
def boxplot_dataframe(df, title):
    """
	Boxplot for all the features sharing the same y axis
	df: pandas dataframe
	title: String, title of the plot
	"""
    results = []
    names = []
    for name in df.columns[:-1]:
        results.append(df[name])
        names.append(name)
    fig = plt.figure(figsize=(11, 6))
    fig.suptitle(title)
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()


def scatter_threshold(df, feature1, feature2, thresh):
    plt.scatter(
        df[str(feature1)][df[str(feature2)] > thresh],
        df[str(feature2)][df[str(feature2)] > thresh],
    )


def compare_y_test_y_pred(y_test, y_pred):
    plt.plot(y_test, y_pred, "r.")
    plt.plot(y_test, y_test, "k-")
    plt.xlabel("y_test")
    plt.ylabel("y_pred")
    plt.title("Scatter plot y_test vs y_pred")


def heatmap_correllation(df, method="pearson"):
    """
	Heatmap correlation for the dataframe 
	df: pandas dataframe
	method: default='pearson'
	"""
    corrmat = df.corr(method=method)
    f, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(corrmat, vmax=1, square=True)


def largest_correlations(df, feature, k):
    """
	giving a correlation heatmap of the best k features correlated to the feature given 
	df: pandas dataframe
	feature: name of the feature 
	k: number of best features
	"""
    corrmat = df.corr()
    cols = corrmat.nlargest(k, str(feature))[str(feature)].index
    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(
        cm,
        cbar=True,
        annot=True,
        square=True,
        fmt=".2f",
        annot_kws={"size": 10},
        yticklabels=cols.values,
        xticklabels=cols.values,
    )
    plt.show()


# Heatmap for categorical data
def heatmap_Cramer_V(df_cut, figsize=(7, 6)):
    """
	Heatmap for categorical data using Cramer's V 
	df_cut: dataframe containing only categorical features
	figsize: Figure size of the plot, default= (7, 6)
	"""

    def cramers_corrected_stat(confusion_matrix):
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

    cols = df_cut.columns.values.tolist()
    corrM = np.zeros((len(cols), len(cols)))
    for col1, col2 in itertools.combinations(cols, 2):
        idx1, idx2 = cols.index(col1), cols.index(col2)
        corrM[idx1, idx2] = cramers_corrected_stat(
            pd.crosstab(df_cut[col1], df_cut[col2])
        )
        corrM[idx2, idx1] = corrM[idx1, idx2]

    corr = pd.DataFrame(corrM, index=cols, columns=cols)
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(corr, annot=True, ax=ax)
    ax.set_title("Cramer V Correlation between the categorical features")


# plotting a 3D scatter plot according to a categorical feature
def plot3D(df, feature, feature1, feature2, feature3):
    fig = px.scatter_3d(
        df,
        x=str(feature1),
        y=str(feature2),
        z=str(feature3),
        color=str(feature),
    )
    fig.show()


# plotting a scatter plot according to a categorical feature
def plot2D(df, feature, feature1, feature2):
    fig = px.scatter_3d(
        df, x=str(feature1), y=str(feature2), color=str(feature)
    )
    fig.show()


def missing_values_percentage(df):
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame(
        {"column_name": df.columns, "percent_missing": percent_missing}
    )
    missing_value_df.sort_values(
        "percent_missing", inplace=True, ascending=False
    )
    missing_value_df["percent_missing"].hist()
    plt.title(
        "histogram for the percentage of missing values for each feature"
    )
    return missing_value_df
