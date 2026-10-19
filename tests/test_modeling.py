import numpy as np
import pandas as pd
import pytest
from credit_risk.modeling import (
    expected_calibration_error, evaluate_probabilities, stratified_split,
)


def test_ece_is_zero_for_perfectly_calibrated_simple_case():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(y, p, n_bins=2) == pytest.approx(0.0)


def test_ece_rejects_invalid_probabilities():
    with pytest.raises(ValueError):
        expected_calibration_error([0, 1], [-0.1, 1.1])


def test_metrics_are_finite_for_nontrivial_predictions():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.7, 0.8, 0.4, 0.6])
    metrics = evaluate_probabilities(y, p)
    assert 0 <= metrics["ROC_AUC"] <= 1
    assert metrics["Brier_score"] >= 0
    assert metrics["Log_loss"] >= 0


def test_split_disjoint_and_stratified():
    X = pd.DataFrame({"x": range(1000)})
    y = pd.Series([0] * 800 + [1] * 200)
    split = stratified_split(X, y)
    assert set(split.X_train.index).isdisjoint(split.X_cal.index)
    assert set(split.X_train.index).isdisjoint(split.X_test.index)
    assert set(split.X_cal.index).isdisjoint(split.X_test.index)
    assert abs(split.y_test.mean() - y.mean()) < 0.03
