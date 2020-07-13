import pandas as pd
import numpy as np

from sklearn.preprocessing import PowerTransformer
from scipy.stats import zscore
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import PolynomialFeatures




def iqr(df):
    """
	filtering a dataframe using the IQR method
	df: dataframe
	returns:
	- dataframe filtered 
	"""
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df_out_IQR = df[
        ~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)
    ]
    print(df_out_IQR.shape, df.shape)
    return df_out_IQR


def z_score(df):
    """
	filtering a dataframe using the zscore method
	df: dataframe
	returns:
	- dataframe filtered 
	"""
    z = np.abs(zscore(df))
    df_out_z = df[(z < 3).all(axis=1)]
    print(df_out_z.shape, df.shape)
    return df_out_z


def selectkbest(df, number, score_function="f_regression"):
    """
	Feature selection using an univariate selection 
	df: pandas dataframe
	number: the number of features you want to select
	score_function : 'f_regression' for regression , 'f_classif' for classification 
	returns: 
	- a dataframe containing features and their respective score
	"""
    bestfeatures = SelectKBest(score_func=score_function, k=number)
    fit = bestfeatures.fit(df.iloc[:, :-1], df.iloc[:, -1])
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(df.iloc[:, :-1].columns)
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ["features", "Score"]
    return featureScores


def transform_data_normal(
    df,
    transformer=PowerTransformer(method="yeo-johnson"),
    liste_prohibited=["Class"],
):
    """
	Transfrom your data in a more gaussian looking density 
	df: pandas dataframe
	transformer : PowerTransformer(method='yeo-johnson') by default , you can choose other transformers like the box cox
	liste_prohibited : list of features that don't need a transformation , by default the feature 'Class'
	returns:
	- dataframe transformed 
	- list of all fitted transformers for each feature that have been transformed
	"""
    list_transformers = list()
    hf = pd.DataFrame()
    for i in df.columns:
        if i not in liste_prohibited:
            transf = transformer.fit(np.array((df[i] + 1)).reshape(-1, 1))
            hf = pd.concat(
                [
                    hf,
                    pd.DataFrame(
                        transf.transform(np.array((df[i] + 1)).reshape(-1, 1)),
                        columns=pd.DataFrame(df[i]).columns,
                    ),
                ],
                axis=1,
            )
            list_transformers.append(transf)
        else:
            hf = pd.concat([hf, df[i]], axis=1)
    return hf, list_transformers


def polynomial_transformation(df, degree=3):
    """
	Transform dataframe using a polynomial transformation 
	df: pandas dataframe
	degree: number of degrees you want to transform , default=3
	returns:
	- dataframe transformed 
	"""
    poly = PolynomialFeatures(degree).fit(df.iloc[:, :-1])
    X_transform = poly.fit_transform(df.iloc[:, :-1])
    poly.get_feature_names(input_features=df.columns[:-1])
    X_transform = pd.DataFrame(
        X_transform,
        columns=poly.get_feature_names(input_features=df.columns[:-1]),
    )
    return X_transform
