import random

import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from timeit import default_timer as timer

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor



def random_search_optimization(
    X_train,
    y_train,
    MAX_EVALS=500,
    N_FOLDS=5,
    param_grid={
        "max_depth": list(range(1, 21)),
        "min_samples_split": list(range(2, 6)),
        "learning_rate": list(
            np.logspace(np.log(0.001), np.log(0.2), base=np.exp(1), num=1000)
        ),
        "subsample": list(
            1
            - np.logspace(np.log(0.001), np.log(0.5), base=np.exp(1), num=1000)
        ),
        "max_features": ["sqrt", "auto", "log2"],
    },
):
    """
	perform a random search optimization for the hyper-parameters 
	X_train
	y_train
	MAX_EVALS : max evaluations, default=500
	param_grid : parameters of the model to use, by default for GradientBoostingRegressor
	returns:
	- dataframe with columns = ['loss', 'params', 'iteration', 'estimators', 'time']
	"""

    def random_objective(params, iteration, n_folds=N_FOLDS):
        start = timer()
        model = GradientBoostingRegressor(
            validation_fraction=0.2, n_iter_no_change=5, tol=0.01, **params
        )
        cv_results = []
        cv_estimators = []
        kfold = KFold(n_splits=N_FOLDS)
        for train_index, test_index in kfold.split(X_train):
            X_train_cv = X_train.iloc[train_index, :]
            Y_train_cv = y_train.iloc[train_index]
            X_test_cv = X_train.iloc[test_index, :]
            Y_test_cv = y_train.iloc[test_index]
            model.fit(X_train_cv, Y_train_cv)
            y_pred = model.predict(X_test_cv)
            variance = np.var(y_pred)
            mse = mean_squared_error(Y_test_cv, y_pred)
            correlation = spearmanr(np.array(Y_test_cv), y_pred)[0]
            scoring = variance * correlation / mse
            cv_results.append(scoring)
            cv_estimators.append(model.n_estimators_)
        end = timer()
        loss = 1 - np.array(cv_results).mean()
        n_estimators = cv_estimators[np.array(cv_results).argmax()]
        return [loss, params, iteration, n_estimators, end - start]

    random_results = pd.DataFrame(
        columns=["loss", "params", "iteration", "estimators", "time"],
        index=list(range(MAX_EVALS)),
    )
    # Iterate through the specified number of evaluations
    for i in range(MAX_EVALS):
        # Randomly sample parameters for gbm
        params = {
            key: random.sample(value, 1)[0]
            for key, value in param_grid.items()
        }
        results_list = random_objective(params, i)
        # Add results to next row in dataframe
        random_results.loc[i, :] = results_list
    random_results.sort_values("loss", ascending=True, inplace=True)
    random_results.reset_index(inplace=True, drop=True)
    return random_results
