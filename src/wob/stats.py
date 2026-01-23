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

    Uses a memory-efficient approach to compute the variance-covariance matrix.

    Parameters:
    mdl : LogisticRegression
        Fitted logistic regression model from sklearn.
    X : np.ndarray
        Training data used to fit the model.

    Returns the Wald statistics and p-values for each coefficient, including the intercept.
    """

    # Get the coefficients and their standard errors
    coefs = mdl.coef_[0]
    intercept = mdl.intercept_[0]
    params = np.append(intercept, coefs)

    # Calculate predicted probabilities
    pred_probs = mdl.predict_proba(X)[:, 1]

    # Construct the design matrix with intercept
    X_design = np.hstack((np.ones((X.shape[0], 1)), X))

    # Calculate the variance-covariance matrix without forming a large diagonal matrix
    weights = pred_probs * (1 - pred_probs)
    XtWX = X_design.T @ (X_design * weights[:, None]) #pylint: disable=C0103

    # Prefer direct inverse; if singular/ill-conditioned, fallback to gentle regularization
    try:
        cov = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        eps = 1e-8 * np.trace(XtWX) / XtWX.shape[0]
        cov = np.linalg.inv(XtWX + np.eye(XtWX.shape[0]) * eps)

    # Standard errors are the square roots of the diagonal elements
    std_errors = np.sqrt(np.diag(cov))

    # Calculate Wald statistics and p-values
    wald_stats = (params / std_errors) ** 2
    p_values = 1 - stats.chi2.cdf(wald_stats, df=1)

    return wald_stats, p_values
