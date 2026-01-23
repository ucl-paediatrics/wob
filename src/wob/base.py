"""Core functions for WOB analysis."""
import polars as pl
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

WOB_SIGNS = [
    'Exhaustion', 
    'Expiratory Noises', 
    'Head Bobbing', 
    'Impending Respiratory Arrest', 
    'Inspiratory Noises', 
    'Intercostal Recession', 
    'Nasal Flaring', 
    'Normal', 
    'Sternal Recession', 
    'Subcostal Recession', 
    'Tracheal Tug'
]

#pylint: disable=too-many-arguments,too-many-positional-arguments
def join_observations_to_outcomes(
    df_obs: pl.DataFrame,
    df_out: pl.DataFrame,
    filter_event_types: tuple[str, ...] = tuple(['2222','unplanned_picu']),
    tolerance_hours: int = 12,
    min_obs_datetime: pl.Datetime | None = None,
    max_obs_datetime: pl.Datetime | None = None
) -> pl.DataFrame:

    """Joins observations to outcomes using a left join. 
    Tolerance can be specified (in hours, defaults to 12)
    """

    df_obs_filtered = df_obs.clone()
    df_out_filtered = df_out.clone()

    if filter_event_types:
        df_out_filtered = df_out_filtered.filter(
            pl.col('event_type').is_in(filter_event_types)
        )

    if max_obs_datetime is not None:
        df_obs_filtered = df_obs_filtered.filter(
            pl.col('measurement_datetime') < max_obs_datetime #type: ignore
        )

    if min_obs_datetime is not None:
        df_obs_filtered = df_obs_filtered.filter(
        pl.col('measurement_datetime') >= min_obs_datetime #type: ignore
        )

    df_join_result = df_obs_filtered.sort(
        ['person_id', 'measurement_datetime']
    ).join_asof(
        df_out_filtered.sort(
            ['person_id', 'event_datetime']
        ),
        tolerance=f'{tolerance_hours}h',
        left_on = 'measurement_datetime',
        right_on = 'event_datetime',
        by = 'person_id',
        strategy = 'forward',
    )

    return df_join_result

def add_time_to_event_columns(df_join: pl.DataFrame) -> pl.DataFrame:
    """Adds time_to_event_hrs and event_within_time_tolerance columns to joined dataframe."""
    df_added = df_join.clone()

    df_added = df_added.with_columns(
        (
            (pl.col('event_datetime') - pl.col('measurement_datetime')).dt.total_seconds()/3600
        ).alias("time_to_event_hrs")
    )

    df_added = df_added.with_columns(
        pl.col('event_type').is_not_null().alias('event_within_time_tolerance')
    )

    return df_added

def threshold_ind_metrics(
    test_targets,
    probabilities
) -> dict:
    """Calculates threshold-independent metrics"""

    measures = {}

    measures['ROC-AUC score'] = metrics.roc_auc_score(test_targets, probabilities)
    measures['Brier score loss'] = metrics.brier_score_loss(test_targets, probabilities)
    measures['Average precision (PR-AUC) score'] = metrics.average_precision_score(test_targets, probabilities)

    return measures

def threshold_dep_metrics(
    test_targets,
    probabilities,
    threshold=0.5
) -> dict:
    """Calculates threshold-dependent metrics"""
    predictions = (probabilities >= threshold).astype(int)

    measures = {}
    measures['Confusion matrix'] = metrics.confusion_matrix(y_pred = predictions, y_true = test_targets)
    measures['Sensitivity (recall)'] = metrics.recall_score(test_targets, predictions)
    measures['Specificity'] = metrics.recall_score(test_targets, predictions, pos_label=0)
    measures['Precision score (PPV)'] = metrics.precision_score(test_targets, predictions)
    measures['F1 score'] = metrics.f1_score(test_targets, predictions)

    return measures

#pylint: disable=invalid-name
def wob_LR(
    train,
    test,
    class_weighting=None,
    selected_signs: list[str] | None = None
) -> dict:
    """Fits a logistic regression model to predict WOB outcome from selected signs.
    
    Arguments:
      train: A polars DataFrame containing training data
      test: A polars DataFrame containing test data
      class_weighting: Class weighting to use in LogisticRegression (default None)
      selected_sign: A sign or list of signs to use as features (default WOB_SIGNS)
    """
    if selected_signs is None:
        selected_signs = WOB_SIGNS # It's better to have mutable default args as None and then assign inside function

    X_train = train.select(selected_signs)

    y_train = train.select(pl.col('event_within_time_tolerance')).to_numpy().ravel()

    X_test = test.select(selected_signs)

    y_test = test.select(pl.col('event_within_time_tolerance')).to_numpy().ravel()

    mdl = LogisticRegression(class_weight=class_weighting)
    mdl.fit(X_train, y_train)

    preds = mdl.predict(X_test)
    probs = mdl.predict_proba(X_test)[:,1]

    return {
        'y_test' : y_test,
        'predictions' : preds,
        'probabilities' : probs,
        'model': mdl
    }

#pylint: disable=invalid-name
def single_sign_LRs(
    train,
    test,
    class_weighting=None
) -> dict[str, dict]:
    """Create and fit logistic regression models for each WOB sign individually."""
    mdls = {sign : wob_LR(train, test, selected_signs=[sign], class_weighting=class_weighting) for sign in WOB_SIGNS}

    return mdls

def pews_scores(data:pl.DataFrame):
    """Calculates the PEWS WOB score as the maximum score of all signs which are present"""
    scores = {'Normal' : 0,
               'Nasal Flaring' : 1,
               'Subcostal Recession' : 1,
               'Head Bobbing' : 2,
               'Tracheal Tug' : 2,
               'Intercostal Recession' : 2,
               'Inspiratory Noises' : 2,
               'Expiratory Noises' : 2,
               'Sternal Recession' : 4,
               'Exhaustion' : 4,
               'Impending Respiratory Arrest' : 4}

    return data.select(
        pl.max_horizontal([
            pl.col(sign).cast(pl.Boolean) * scores[sign]
            for sign in WOB_SIGNS
        ])).to_numpy().ravel()


def pews_model(data: pl.DataFrame) -> dict:
    """Calculate PEWS score and outcome
    
    Returns:
      A dictionary with two keys:
      "scores": A numpy vector of scores as integers 
      "outcomes": A binary label vector
    
    """
    scores = pews_scores(data)
    outcomes = data.select(pl.col('event_within_time_tolerance')).to_numpy().ravel()

    return {
        'scores' : scores,
        'outcomes' : outcomes
    }
