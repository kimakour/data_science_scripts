import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from imblearn.over_sampling import SMOTENC


def resample_smotenc(df, maximum=53):
    df["class"] = np.where(df["OUTPUT"] < maximum, 0, 1)
    df_pre_sample = pd.concat(
        [
            df.select_dtypes(include="int64").iloc[:, :-1],
            df.select_dtypes(include="float64"),
            df["class"],
        ],
        1,
    )
    smote_nc = SMOTENC(
        categorical_features=list(
            range(
                (
                    df_pre_sample.shape[1]
                    - df_pre_sample.select_dtypes(include="float64").shape[1]
                    - 1
                )
            )
        ),
        random_state=0,
    )
    X_resampled, y_resampled = smote_nc.fit_resample(
        df_pre_sample.iloc[:, :-1], df_pre_sample.iloc[:, -1]
    )
    df_resampled = pd.concat([X_resampled, y_resampled], 1)
    return df_resampled


def crossvalidation_oversampling_mixed_data(df):
    pipelines = []
    pipelines.append(
        (
            "ScaledLR",
            Pipeline(
                [("Scaler", StandardScaler()), ("LR", LinearRegression())]
            ),
        )
    )
    pipelines.append(
        (
            "ScaledLASSO",
            Pipeline([("Scaler", StandardScaler()), ("LASSO", Lasso())]),
        )
    )
    pipelines.append(
        (
            "ScaledEN",
            Pipeline([("Scaler", StandardScaler()), ("EN", ElasticNet())]),
        )
    )
    pipelines.append(
        (
            "ScaledKNN",
            Pipeline(
                [("Scaler", StandardScaler()), ("KNN", KNeighborsRegressor())]
            ),
        )
    )
    pipelines.append(("GBM", GradientBoostingRegressor()))
    pipelines.append(("XGB", XGBRegressor()))
    pipelines.append(("RF", RandomForestRegressor()))
    pipelines.append(("LGBM", LGBMRegressor()))
    pipelines.append(
        ("SVR", Pipeline([("Scaler", StandardScaler()), ("SVR", SVR())]))
    )

    results = []
    results_resample = []
    names = []

    for name, model in pipelines:
        cv_results_resample = []
        cv_results = []
        kfold = KFold(n_splits=5, random_state=21)
        for train_index, test_index in kfold.split(df):
            df_resampled = resample_smotenc(
                pd.concat(
                    [df.iloc[train_index, :-1], df.iloc[train_index, -1]], 1
                )
            )
            model.fit(df_resampled.iloc[:, :-2], df_resampled.iloc[:, -2])
            cv_results_resample.append(
                mean_squared_error(
                    df.iloc[test_index, -1],
                    model.predict(df.iloc[test_index, :-1]),
                )
            )
            model.fit(df.iloc[train_index, :-1], df.iloc[train_index, -1])
            cv_results.append(
                mean_squared_error(
                    df.iloc[test_index, -1],
                    model.predict(df.iloc[test_index, :-1]),
                )
            )

        results.append(cv_results)
        results_resample.append(cv_results_resample)
        names.append(name)
    fig, ax = plt.subplots(1, 2, sharey=True)
    fig.set_size_inches(15, 10)
    ax[0].boxplot(results)
    ax[0].set_xticklabels(names)
    ax[0].set_title("normal")
    ax[1].boxplot(results_resample)
    ax[1].set_xticklabels(names)
    ax[1].set_title("oversampling")
    fig.suptitle(
        "Model Comparison for the mean squared error without/with resampling"
    )
    return names, results, results_resample
