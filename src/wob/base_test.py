"""Tests for src/base.py module using pytest."""
import numpy as np
import polars as pl

from wob import base
from wob.base import pews_scores


def make_empty_row():
    """Helper that returns a dict with all WOB sign columns set to 0."""

    return {sign: 0 for sign in base.WOB_SIGNS}


def test_pews_all_zero():
    """Test that a row with no signs present gets a PEWS score of 0."""
    row = make_empty_row()
    df = pl.DataFrame([row])

    scores = pews_scores(df)

    np.testing.assert_array_equal(scores, np.array([0]))


def test_pews_single_sign_nasal_flaring():
    """Test that a row with only Nasal Flaring present gets a PEWS score of 1."""
    row = make_empty_row()
    row['Nasal Flaring'] = 1
    df = pl.DataFrame([row])

    scores = pews_scores(df)

    np.testing.assert_array_equal(scores, np.array([1]))


def test_pews_single_sign_head_bobbing():
    """Test that a row with only Head Bobbing present gets a PEWS score of 2."""
    row = make_empty_row()
    row['Head Bobbing'] = 1
    df = pl.DataFrame([row])

    scores = pews_scores(df)

    np.testing.assert_array_equal(scores, np.array([2]))


def test_pews_multiple_rows_and_max_selection():
    """Test that multiple rows are handled and max score is selected."""
    # Row1: nasal flaring (1) -> score 1
    # Row2: sternal recession (1) and nasal flaring (1) -> max score 4
    row1 = make_empty_row()
    row1['Nasal Flaring'] = 1

    row2 = make_empty_row()
    row2['Nasal Flaring'] = 1
    row2['Sternal Recession'] = 1

    df = pl.DataFrame([row1, row2])

    scores = pews_scores(df)

    np.testing.assert_array_equal(scores, np.array([1, 4]))


def test_pews_all_signs_present():
    """All WOB signs present should produce the maximum mapped score (4)."""
    row = make_empty_row()
    for sign in base.WOB_SIGNS:
        row[sign] = 1

    df = pl.DataFrame([row])

    scores = pews_scores(df)

    np.testing.assert_array_equal(scores, np.array([4]))


def test_pews_conflicting_signs_max():
    """When lower- and higher-scored signs are present, the max should win."""
    row = make_empty_row()
    row['Head Bobbing'] = 1
    row['Exhaustion'] = 1
    df = pl.DataFrame([row])

    np.testing.assert_array_equal(pews_scores(df), np.array([4]))


def test_pews_non_binary_values():
    """Indicator values are binary; non zero values should all be treated as a 1."""
    row = make_empty_row()
    row['Nasal Flaring'] = 3
    df = pl.DataFrame([row])

    # nasal flaring maps to 1 regardless of indicator value
    np.testing.assert_array_equal(pews_scores(df), np.array([1]))


def test_pews_normal_only():
    """'Normal' maps to 0 even when present."""
    row = make_empty_row()
    row['Normal'] = 1
    df = pl.DataFrame([row])

    np.testing.assert_array_equal(pews_scores(df), np.array([0]))
