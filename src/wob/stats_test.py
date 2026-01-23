"""Test code for custom stats module."""
import numpy as np
from scipy import stats
from sklearn.datasets import make_sparse_uncorrelated
from sklearn.linear_model import LogisticRegression
from statsmodels.discrete.discrete_model import Logit
import statsmodels.api as sm

from wob.stats import wald_test


def test_wald_test():
    """Test the wald_test function with a simple logistic regression model.
    
    This just checks that the function runs and returns outputs of expected shape.
    
    """

    # Create a simple dataset
    X = np.array([[0], [1], [2], [3], [4], [5]])
    y = np.array([0, 0, 0, 1, 1, 1])

    # Fit a logistic regression model
    mdl = LogisticRegression()
    mdl.fit(X, y)

    # Perform Wald test
    wald_stats, p_values = wald_test(mdl, X)

    # Check that the outputs are of correct length
    assert len(wald_stats) == X.shape[1] + 1  # +1 for intercept
    assert len(p_values) == X.shape[1] + 1

    # Check that p-values are between 0 and 1
    assert all(0 <= p <= 1 for p in p_values)

def test_wald_against_statsmodel():
    """Tests wald_test function against statsmodels implementation."""
    # Create a sample dataset with a binary outcome that is correlated with some of the features - but not all.
    X, y = make_sparse_uncorrelated(
        n_samples=10000,
        n_features=10,
        random_state=42,
    )
    y = (y > 0.5).astype(int)  # Convert to binary outcome

    # Fit a logistic regression model using sklearn
    skl_mdl = LogisticRegression(C=np.inf, fit_intercept=True)
    skl_mdl.fit(X, y)

    # Perform Wald test using custom function
    wald_stats_custom, p_values_custom = wald_test(skl_mdl, X)

    # Fit a logistic regression model using statsmodels. Need to add constant for intercept.
    logit_model = Logit(y, sm.add_constant(X))
    sm_result = logit_model.fit(disp=0)

    # Get Wald statistics and p-values from statsmodels
    wald_stats_sm = (sm_result.params / sm_result.bse) ** 2
    p_values_sm = 1 - stats.chi2.cdf(wald_stats_sm, df=1)

    # The solvers may differ slightly, so allow for some tolerance in comparisons. 2% seems reasonable.
    relative_tolerance = 0.02

    # Compare intercepts and coefficients
    np.testing.assert_allclose(skl_mdl.intercept_[0], sm_result.params[0], rtol=relative_tolerance)
    np.testing.assert_allclose(skl_mdl.coef_[0], sm_result.params[1:], rtol=relative_tolerance)
    # Compare results
    np.testing.assert_allclose(wald_stats_custom, wald_stats_sm, rtol=relative_tolerance)
    np.testing.assert_allclose(p_values_custom, p_values_sm, rtol=relative_tolerance)
    print(skl_mdl.coef_.ravel())
