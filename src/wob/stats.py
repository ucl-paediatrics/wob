"""Specialised stats-related functions for WOB project."""
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression

def wald_test(
    mdl: LogisticRegression,
    X: np.ndarray,
):
    """
    Perform Wald test for the significance of coefficients in a logistic regression model. 

    Parameters:
    mdl : LogisticRegression
        Fitted logistic regression model from sklearn.
    X : np.ndarray
        Training data used to fit the model.
    """

    # Get the coefficients and their standard errors
    coefs = mdl.coef_[0]
    intercept = mdl.intercept_[0]
    params = np.append(intercept, coefs)

    # Calculate predicted probabilities
    pred_probs = mdl.predict_proba(X)[:, 1]

    # Construct the design matrix with intercept
    X_design = np.hstack((np.ones((X.shape[0], 1)), X))

    # Calculate the variance-covariance matrix
    V = np.diag(pred_probs * (1 - pred_probs)) #pylint: disable=C0103
    XtVX_inv = np.linalg.inv(X_design.T @ V @ X_design) #pylint: disable=C0103

    # Standard errors are the square roots of the diagonal elements
    std_errors = np.sqrt(np.diag(XtVX_inv))

    # Calculate Wald statistics and p-values
    wald_stats = (params / std_errors) ** 2
    p_values = 1 - stats.chi2.cdf(wald_stats, df=1)

    return wald_stats, p_values
