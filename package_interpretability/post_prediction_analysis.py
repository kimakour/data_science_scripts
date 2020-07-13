import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from scipy.integrate import simps
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler



def compute_adjusted_cosine_similarity(
    feature_importances, X_train, X_test, y_test, list_indexes
):
    """
	Computing an adjusted cosine similarity between two dataframes
	feature_importances : the feature importance score for each feature 
	X_train : pandas dataframe/series 
	X_test : pandas dataframe/series
	list_indexes : list of indexes for the X_test that we want to know their similarity with instances in X_train
	returns:
	- pandas dataframe containing the cosine similarity for each vector between X_train and X_test
	"""
    mmsc = MinMaxScaler().fit(X_train)
    a = np.multiply(feature_importances, mmsc.transform(X_train))
    b = np.multiply(feature_importances, mmsc.transform(X_test))
    return pd.DataFrame(
        cosine_similarity(a, b),
        columns=y_test.loc[list_indexes, "y_test"].to_list(),
    )


def REC_curve(y_true, y_pred, Begin_Range=0, End_Range=80, Interval_Size=1):
    """
	Computing the Regression Error Characteristic for a given model that has already predicted the X_test
	y_true : y_test , pandas dataframe
	y_pred : the result of model.predict(X_test), generally a numpy array 
	Begin_Range: the minimum error 
	End_Range: the maximum error
	Interval_Size: the step to compute the area under the curve
	"""

    def REC(y_true, y_pred, Begin_Range, End_Range, Interval_Size):
        Accuracy = []
        Begin_Range = 0
        End_Range = 84
        Interval_Size = 1
        Epsilon = np.arange(Begin_Range, End_Range, Interval_Size)
        for i in range(len(Epsilon)):
            count = 0.0
            for j in range(len(y_true)):
                if np.abs(y_true[j] - y_pred[j]) < Epsilon[i]:
                    count = count + 1
            Accuracy.append(count / len(y_true))
        AUC = simps(Accuracy, Epsilon) / End_Range
        return Epsilon, Accuracy, AUC

    # finding the deviation and accuracy, and area under curve for plotting
    Deviation, Accuracy, AUC = REC(
        np.array(y_true),
        np.array(y_pred),
        Begin_Range,
        End_Range,
        Interval_Size,
    )
    # Calculating R^2 of the test and predicted values
    RR = r2_score(np.array(y_true), y_pred)
    # Plotting
    plt.figure(figsize=(14, 8))
    plt.suptitle("REC & Residuals")
    plt.subplot(1, 2, 1)
    plt.title("Residuals")
    plt.scatter(y_true, y_pred, color="darkorange")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.plot(
        [y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "k--", lw=4
    )
    plt.text(45, -5, r"$R^2 = %0.4f$" % RR, fontsize=15)
    plt.subplot(1, 2, 2)
    plt.title("Regression Error Characteristic (REC)")
    plt.plot(Deviation, Accuracy, "--b", lw=3)
    plt.xlabel("Deviation")
    plt.ylabel("Accuracy (%)")
    plt.text(1.1, 0.07, "AUC = %0.4f" % AUC, fontsize=15)

    plt.show()
