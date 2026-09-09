import numpy as np

from pointcept.utils.eval_cm import compute_prediction_integrity


def test_prediction_integrity_excludes_background_predictions():
    label = np.array([12, 12, 12, 12, 12, 24, 24, 0, 0], dtype=np.uint8)
    pred = np.array([0, 0, 12, 12, 11, 0, 0, 12, 24], dtype=np.uint8)

    result = compute_prediction_integrity(pred, label)

    assert np.isclose(result["per_class_integrity"][12], 2 / 3)
    assert result["per_class_dominant_pred"][12] == 12
    assert result["per_class_fragment_counts"][12] == 2
    assert result["per_class_pred_distribution"][12] == {11: 1, 12: 2}
    assert result["per_class_recognized_points"][12] == 3

    assert np.isnan(result["per_class_integrity"][24])
    assert result["per_class_dominant_pred"][24] is None
    assert result["per_class_fragment_counts"][24] == 0
    assert result["per_class_pred_distribution"][24] == {}
    assert result["per_class_recognized_points"][24] == 0

    assert np.isclose(result["mean_integrity"], 2 / 3)
    assert np.isclose(result["weighted_mean_integrity"], 2 / 3)
    assert np.isclose(result["global_integrity"], 2 / 3)
