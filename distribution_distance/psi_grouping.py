import numpy as np
import matplotlib.pyplot as plt

import networkx as nx

from kl_divergence import divide_density_to_intervals, psi


def weighted_psi_grouping(
    df, column, list_important_features, feature_importance, threshold=0.25
):
    """
    group levels for a categorical feature accroding to the distribution of the
    other features towards them.

    Arguments:
    df : pandas dataframe
    column : the categorical feature we want to study
    list_important_features : the features we want to compare their distribution
    feature_importance : the weight to give to each feature
    threshold : threshold for the psi value , default=0.25

    returns:
    dictionnary : key = level of the categorical feature , value = level for the categorical
    feature 
    The dictionnary is a dictionnary of correspondance between levels (levels who have
    a weighted psi value less than 0.25 are put together)

    """
    dictionnary_psi = dict()
    for level in df[column].unique():
        subdf = df[df[column] == level]
        if len(subdf) < 10:
            continue
        other_levels = list(set(df[column].unique().tolist()) - set([level]))
        for other_level in other_levels:
            weighted_psi = 0
            subdf_other = df[df[column] == other_level]
            if len(subdf_other) < 10:
                continue
            for index, important_feature in enumerate(list_important_features):
                arrays = divide_density_to_intervals(
                    subdf[important_feature], subdf_other[important_feature]
                )
                weighted_psi = (
                    weighted_psi
                    + psi(arrays[0], arrays[1]) * feature_importance[index]
                )
            if weighted_psi < threshold:
                dictionnary_psi[level] = other_level
    return dictionnary_psi


def draw_network_graph(dicto):
    """
    Plot a graph of correspondance between levels of a categorical feature that have
    the same distribution towards given features 
    """
    liste = list()
    dicto = dictionnary_psi
    for i in dicto.keys():
        if isinstance(dicto[i], list):
            for j in dicto[i]:
                liste.append({"start": i, "end": j})
        else:
            liste.append({"start": i, "end": dicto[i]})
    G = nx.Graph()
    for pts in liste:
        G.add_edge(pts["start"], pts["end"])
    pos = nx.spring_layout(G, k=0.2)
    nx.draw(G, pos, node_color="lawngreen", with_labels=True, k=20)
