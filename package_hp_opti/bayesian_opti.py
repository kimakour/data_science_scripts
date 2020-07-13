import csv

import numpy as np

from timeit import default_timer as timer
from scipy.stats import spearmanr


from hyperopt import hp, tpe, Trials, fmin, STATUS_OK

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor



def bayesian_optimization(
    X_train,
    y_train,
    n_folds=5,
    outpath="rgbm_trials_bayesian.csv",
    MAX_EVALS=500,
    space={
        "max_features": hp.quniform("max_features", 1, 9, 1),
        "learning_rate": hp.loguniform(
            "learning_rate", np.log(0.001), np.log(0.2)
        ),
        "max_depth": hp.quniform("max_depth", 1, 21, 1),
        "min_samples_split": hp.quniform("min_samples_split", 2, 6, 1),
        "subsample": 1
        - hp.loguniform("subsample", np.log(0.001), np.log(0.5)),
    },
):
    """
	Bayesian optimization for searching the best set of hyper-parameters 
	X_train
	y_train
	n_folds: number of folds for crossvalidation
	outpath: the path of the csv folder for recording the history of the optimization, default='rgbm_trials_bayesian.csv'
	MAX_EVALS: number of evaluations
	space: dictionnary containing the space of hyper parameters to search
	"""

    tpe_algorithm = tpe.suggest
    bayes_trials = Trials()
    MAX_EVALS = MAX_EVALS
    out_file = outpath
    of_connection = open(out_file, "w")
    writer = csv.writer(of_connection)
    # Write the headers to the file
    writer.writerow(
        ["loss", "params", "iteration", "estimators", "train_time"]
    )
    of_connection.close()

    def objective(params, n_folds=n_folds):
        global ITERATION
        ITERATION += 1
        # Make sure parameters that need to be integers are integers
        for parameter_name in [
            "max_features",
            "min_samples_split",
            "max_depth",
        ]:
            params[parameter_name] = int(params[parameter_name])

        start = timer()
        cv_results = []
        cv_estimators = []
        model = GradientBoostingRegressor(
            validation_fraction=0.2, n_iter_no_change=5, tol=0.01, **params
        )
        kfold = KFold(n_splits=n_folds)
        for train_index, test_index in kfold.split(X_train):
            X_train_cv = X_train.iloc[train_index, :]
            Y_train_cv = y_train.iloc[train_index]
            X_test_cv = X_train.iloc[test_index, :]
            Y_test_cv = y_train.iloc[test_index]
            model.fit(X_train_cv, Y_train_cv)
            y_pred = model.predict(X_test_cv)
            variance = np.var(y_pred)
            correlation = spearmanr(np.array(Y_test_cv), y_pred)[0]
            mse = mean_squared_error(Y_test_cv, y_pred)
            scoring = variance * correlation / mse
            cv_results.append(scoring)
            cv_estimators.append(model.n_estimators_)

        run_time = timer() - start
        loss = 1 - np.array(cv_results).mean()
        n_estimators = cv_estimators[np.array(cv_results).argmax()]

        # Write to the csv file ('a' means append)
        of_connection = open(out_file, "a")
        writer = csv.writer(of_connection)
        writer.writerow([loss, params, ITERATION, n_estimators, run_time])
        # Dictionary with information for evaluation
        return {
            "loss": loss,
            "params": params,
            "iteration": ITERATION,
            "estimators": n_estimators,
            "train_time": run_time,
            "status": STATUS_OK,
        }

    global ITERATION
    ITERATION = 0
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=MAX_EVALS,
        trials=bayes_trials,
    )
