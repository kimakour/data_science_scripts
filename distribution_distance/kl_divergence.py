import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from math import log2, sqrt


def divide_density_to_intervals(
    df1, df2, intervals=100, division_method="quantiles"
):
    """
	Divide the first array to 100 intervals using quantiles 
	Compute the proba for each dataframe (df1 and df2) to be in each interval.

	Arguments: 
	df1 : first dataframe (on which we choose the intervals)
	df2 : second dataframe that we need to compare with the first one
    intervals : number of intervals, default=100
    division_method : how to divide intervals, default= 'quantiles'
    possible methods : ['quantiles', 'bins']

	returns two numpy arrays containing these probabilities

	"""
    if division_method == "quantiles":
        array = pd.qcut(df1, intervals, duplicates="drop", retbins=True)[1]
    elif division_method == "bins":
        array = pd.cut(df1, intervals, duplicates="drop", retbins=True)[1]

    array1 = np.array(
        [
            df1.between(left=array[i], right=array[i + 1]).sum()
            for i in range(len(array) - 1)
        ]
    )
    new_array1 = array1 / array1.sum()
    array2 = np.array(
        [
            df2.between(left=array[i], right=array[i + 1]).sum()
            for i in range(len(array) - 1)
        ]
    )
    if  np.all(array2==0):
        new_array2 = np.where(array2 == 0, 0.000001, array2)
    else:
        new_array2 = array2 / array2.sum()
        np.where(new_array2 == 0, 0.000001, new_array2)
    np.where(new_array1 == 0, 0.000001, new_array1)
    return new_array1, new_array2


def kl_divergence(p, q):
    """
	compute KL divergence for two numpy arrays containing probabilities
	"""
    p[p == 0.0] = 0.00001
    q[q == 0.0] = 0.00001
    return sum(p[i] * log2(p[i] / q[i]) for i in range(len(p)))


def js_divergence(p, q):
    """
	compute Jensen Shannon divergence between two numpy arrays containing probabilities
	"""
    middle = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, middle) + 0.5 * kl_divergence(q, middle)


def js_distance(p, q):
    """
	compute Jensen Shannon distance between two numpy arrays containing probabilites 
	"""
    return sqrt(js_divergence(p, q))


def psi(p, q):
    """
	compute the psi value between two numpy arrays containing probabilities
	"""
    p[p == 0.0] = 0.00001
    q[q == 0.0] = 0.00001
    return kl_divergence(p, q) + kl_divergence(q, p)


def plot_psi_values_dataset(df, percentage=0.66):
    """
    Plot PSI value for each feature in a dataset 
    df: pandas dataframe
    percentage: the percentage to divide the dataframe on(the dataframe must be ordered by
    date) 

    """
    psi_bins = list()
    psi_quantiles = list()
    df.index = list(range(len(df)))
    rows = round(len(df) * percentage)
    for column in df.columns:

        arrays_bins = divide_density_to_intervals(
            df.loc[:rows, column],
            df.loc[rows:, column],
            intervals=100,
            division_method="bins",
        )
        arrays_quantiles = divide_density_to_intervals(
            df.loc[:rows, column], df.loc[rows:, column]
        )

        psi_bins.append(psi(arrays_bins[0], arrays_bins[1]))
        psi_quantiles.append(psi(arrays_quantiles[0], arrays_quantiles[1]))

    bars = list(df.columns)
    y_pos = np.arange(len(bars))

    height_bins = psi_bins
    height_quantiles = psi_quantiles

    fig, axs = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
    axs[0].bar(y_pos, height_quantiles)
    axs[0].axhline(y=0.25, color="r")
    axs[0].set_xticks(y_pos)
    axs[0].set_xticklabels(bars, rotation=70)
    axs[0].set_title("using quantiles")
    axs[1].bar(y_pos, height_bins)
    axs[1].axhline(y=0.25, color="r")
    axs[1].set_xticks(y_pos)
    axs[1].set_xticklabels(bars, rotation=70)
    axs[1].set_title("using bins")
    fig.suptitle("PSI value for the features")
