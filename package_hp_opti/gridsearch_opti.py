from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest


def model_optimisation_gridsearch(
    X_train, y_train, model, params, score="roc_auc", cv=5, n_jobs=-2
):
    """
	Hyper-parameter optimization using gridsearch 
	X_train
	y_train
	model : model to use for the grid search 
	params: dictionary containing the hyper-parameters of the model
	score: scoring for the ML task, default='roc_auc'
	cv: number of folds for crossvalidation, default=5
	n_jobs: number of processors for the grid seach, default=-2 (all of them except one)
	returns:
	- the best estimator 

	"""
    grid_search = GridSearchCV(
        model,
        param_grid=params,
        cv=cv,
        scoring=score,
        verbose=4,
        n_jobs=n_jobs,
    )
    grid_search.fit(X_train, y_train)
    print(grid_search.best_score_)
    print(grid_search.best_params_)
    return grid_search.best_estimator_


def model_opti_grid_search_kbest(
    X_train,
    y_train,
    tuple_model=("RF", RandomForestClassifier()),
    dicto_parameters={
        "kbest__k": [5, 6, 7, 8, 9, 10, 11, 12],
        "RF__n_estimators": [80, 120],
        "RF__max_features": ["sqrt", "log2", None],
    },
):
    """
	hyper-parameter optimization + feature selection using grid search
	"""
    kbest = SelectKBest()
    pipeline = Pipeline([("kbest", kbest), tuple_model])
    grid_search = GridSearchCV(pipeline, dicto_parameters)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_score_)
    print(grid_search.best_params_)
    return grid_search.best_estimator_
