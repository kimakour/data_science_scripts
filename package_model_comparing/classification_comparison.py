# Created libraries
from package_preprocessing.transformations import (
    transform_data_normal,
    scaling,
)


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

# ML basics
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
)
from sklearn.feature_selection import RFE, SelectKBest
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# Data
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns


def evaluation_classification_models(df, nbfolds=10, scoring="roc_auc"):
    X_train, X_test, y_train, y_test = train_test_split(
        df.iloc[:, :-1], df.iloc[:, -1], test_size=0.25, random_state=42
    )
    models = []
    models.append(("LR", LogisticRegression()))
    models.append(("LDA", LinearDiscriminantAnalysis()))
    models.append(("KNN", KNeighborsClassifier()))
    models.append(("CART", DecisionTreeClassifier()))
    models.append(("NB", GaussianNB()))
    models.append(("RF", RandomForestClassifier()))
    models.append(("XGB", XGBClassifier()))
    # evaluate each model in turn
    names = []
    results = []
    for name, model in models:
        kfold = KFold(n_splits=nbfolds)
        if name in ["XGB", "RF"]:
            cv_results = cross_val_score(
                model, X_train, y_train, cv=kfold, scoring=scoring
            )
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)
        elif name in ["LR", "LDA", "NB"]:
            df_norm = transform_data_normal(X_train)[0]
            cv_results = cross_val_score(
                model, df_norm, y_train, cv=kfold, scoring=scoring
            )
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        else:
            df_scaled = scaling(X_train, MinMaxScaler(feature_range=[0, 1]))[0]
            cv_results = cross_val_score(
                model, df_scaled, y_train, cv=kfold, scoring=scoring
            )
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    # boxplot algorithm comparison
    fig = plt.figure(figsize=(11, 6))
    fig.suptitle("Model Comparison")
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()
    return X_test, y_test


def show_classification_results(classifier, x_test, y_test):
    y_pred_test = classifier.predict(x_test)
    report = classification_report(y_true=y_test, y_pred=y_pred_test)
    matrix = confusion_matrix(y_true=y_test, y_pred=y_pred_test)
    print("Test Set:")
    print(report)
    df_cm = pd.DataFrame(
        matrix, index=[i for i in "01"], columns=[i for i in "01"]
    )
    plt.figure(figsize=(7, 5))
    plt.xlabel("Real class")
    plt.ylabel("Predicted class")
    ax = sns.heatmap(df_cm, annot=True, color="blue")
    ax.set(xlabel="real class", ylabel="predicted class")


def show_result_hard_voting(df):
    def hard_voting(df):
        X_train, X_test, y_train, y_test = train_test_split(
            df.iloc[:, :-1], df.iloc[:, -1], test_size=0.25, random_state=42
        )
        models = []
        models.append(("LR", LogisticRegression()))
        models.append(("LDA", LinearDiscriminantAnalysis()))
        models.append(("KNN", KNeighborsClassifier()))
        models.append(("CART", DecisionTreeClassifier()))
        models.append(("NB", GaussianNB()))
        models.append(("RF", RandomForestClassifier()))
        models.append(("XGB", XGBClassifier()))
        # evaluate each model in turn
        results = []
        names = []
        y_predicted = []
        for name, model in models:
            if name in ["XGB", "RF", "CART"]:
                y_predicted.append(model.fit(X_train, y_train).predict(X_test))
            elif name in ["LR", "LDA", "NB"]:
                X_train_norm = transform_data_normal(X_train)[0]
                X_test_norm = transform_data_normal(X_test)[0]
                y_predicted.append(
                    model.fit(X_train_norm, y_train).predict(X_test_norm)
                )

            else:
                X_train_scaled = scaling(
                    X_train, MinMaxScaler(feature_range=[0, 1])
                )[0]
                X_test_scaled = scaling(
                    X_test, MinMaxScaler(feature_range=[0, 1])
                )[0]
                y_predicted.append(
                    model.fit(X_train_scaled, y_train).predict(X_test_scaled)
                )
        return y_predicted, y_test

    def most_frequent(List):
        return max(set(List), key=List.count)

    def outputing_scores(dicto, y_test):
        y_vote = []
        for name, y_predict in dicto.items():
            print(
                "f1 score for the model ",
                name,
                "is: ",
                f1_score(y_test, y_predict),
            )
        for i in range(len(y_test)):
            a = most_frequent(
                list(
                    (
                        dicto["LR"][i],
                        dicto["LDA"][i],
                        dicto["KNN"][i],
                        dicto["CART"][i],
                        dicto["NB"][i],
                        dicto["RF"][i],
                        dicto["XGB"][i],
                    )
                )
            )
            y_vote.append(a)
        print(
            "f1 score for the voting model is: ",
            f1_score(y_test, np.array(y_vote)),
        )

    y_predicted, y_test = hard_voting(df.iloc[:, 4:])
    dicto = dict()
    for i, j in enumerate(y_predicted):
        dicto[names[i]] = j

    outputing_scores(dicto, y_test)


def select_best_features_best_model_recursive_elimination(
    df, name, model, kfold=5, score="accuracy"
):
    for i in range(len(df.columns) - 1):
        rfe = RFE(model, i + 1)
        fit = rfe.fit(df.iloc[:, :-1], df.iloc[:, -1])
        dfe = fit.transform(df.iloc[:, :-1])
        mask = fit.support_
        new_features = []
        for bool, feature in zip(mask, df.columns[:-1]):
            if bool:
                new_features.append(feature)
            cv_results = cross_val_score(
                model, dfe, df.iloc[:, -1], cv=kfold, scoring=score
            )
            print(
                name,
                " /features : ",
                new_features,
                cv_results.mean(),
                cv_results.std(),
            )


def evaluation_pca(df, components):
    X_train, X_test, y_train, y_test = train_test_split(
        df.iloc[:, :-1], df.iloc[:, -1], test_size=0.25, random_state=42
    )
    pca = PCA(n_components=components)
    X_reduced = pca.fit_transform(X_train)
    df_y = pd.DataFrame(y_train)
    df_y.index = range(len(df_y))
    df_pca = pd.concat([pd.DataFrame(X_reduced), df_y], 1)
    a, b = evaluation_classification_models(df_pca)
    return pca, X_test, y_test


def select_best_features_best_model_univariate(df, number):
    def select_best_features(df, number):
        h = SelectKBest(k=number).fit(df.iloc[:, :-1], df.iloc[:, -1])
        df2 = h.transform(df.iloc[:, :-1])
        mask = h.get_support()
        new_features = []
        for bool, feature in zip(mask, df.columns[:-1]):
            if bool:
                new_features.append(feature)
        final_select_df = pd.concat(
            [
                pd.DataFrame(df2, columns=new_features),
                pd.DataFrame(df.iloc[:, -1]),
            ],
            1,
        )
        evaluation_classification_models(final_select_df)

    for i in range(len(df.columns) - 1):
        print("\n\n")
        print("Select ", i + 1, " Best")
        select_best_features(df, i + 1)
