import numpy as np

from pointcept.utils.eval_cm import (
    compute_prediction_integrity,
    eval_volume,
    foreground_confusion_metrics,
    reduce_records,
    summarize_foreground_confusion,
)


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


def _confusion(rows):
    cm = np.zeros((25, 25), dtype=np.int64)
    for gt, pred_counts in rows.items():
        for pred, count in pred_counts.items():
            cm[gt, pred] = count
    return cm


def test_foreground_confusion_metrics_distinguishes_coherent_wrong_and_leakage(
):
    cm = _confusion({
        1: {
            1: 90,
            0: 10
        },  # coherent and correct, with a background miss
        2: {
            3: 80,
            0: 20
        },  # coherent, but assigned to the wrong rib
        3: {
            3: 45,
            4: 45,
            0: 10
        },  # recognized points leak to another rib
    })
    metrics = foreground_confusion_metrics(cm)

    assert metrics["fg_purity_cw"][0] is None
    assert metrics["fg_purity_cw"][1] == 1.0
    assert metrics["fg_recall_cw"][1] == 1.0
    assert metrics["fg_recognition_coverage_cw"][1] == 0.9
    assert metrics["fg_purity_cw"][2] == 1.0
    assert metrics["fg_recall_cw"][2] == 0.0
    assert metrics["fg_purity_cw"][3] == 0.5
    assert metrics["fg_fragment_count_5pct_cw"][3] == 2


def test_foreground_confusion_metrics_absent_missed_threshold_and_diffuse():
    cm = _confusion({
        1: {
            0: 7
        },  # present but completely unrecognized
        2: {
            2: 95,
            3: 5
        },
        3: {
            **{
                pred: 1
                for pred in range(1, 25)
            }, 0: 1
        },
    })
    metrics = foreground_confusion_metrics(cm)

    assert metrics["fg_purity_cw"][1] is None
    assert metrics["fg_recall_cw"][1] is None
    assert metrics["fg_fragment_count_5pct_cw"][1] is None
    assert metrics["fg_recognition_coverage_cw"][1] == 0.0
    assert metrics["fg_fragment_count_5pct_cw"][2] == 2  # exact 5% counts
    assert metrics["fg_fragment_count_5pct_cw"][3] == 0  # 24/25 are under 5%
    assert metrics["fg_purity_cw"][4] is None
    assert metrics["fg_recognition_coverage_cw"][4] is None


def test_foreground_confusion_summary_is_volume_pair_weighted_and_json_safe():
    cm_a = _confusion({1: {1: 100}, 2: {0: 10}})
    cm_b = _confusion({1: {1: 1, 2: 1}, 2: {2: 1}})
    records = {
        "a": foreground_confusion_metrics(cm_a),
        "b": foreground_confusion_metrics(cm_b),
        "old": {
            "mean_integrity": 1.0
        },
    }
    summary = summarize_foreground_confusion(records)

    # Three scored pairs: a/1, b/1 and b/2.  a/2 is unrecognized; absent
    # anatomical ribs are counted separately rather than scored as failures.
    assert summary["fg_scored_pair_count"] == 3
    assert summary["fg_unrecognized_pair_count"] == 1
    assert summary["fg_absent_pair_count"] == 44
    assert np.isclose(summary["fg_purity_mean"], (1 + .5 + 1) / 3)
    assert np.isclose(summary["fg_purity_lt95_fraction"], 1 / 3)
    assert summary["fg_purity_lt95_volume_count"] == 1
    assert np.isclose(summary["fg_recognition_coverage_mean"], .75)
    assert all(value is None or isinstance(value, (int, float))
               for value in summary.values())


def test_eval_volume_rib_guard_and_reduce_preserves_volume_pair_summary():
    pred = np.array([1, 1, 2, 0], dtype=np.int64)
    label = np.array([1, 1, 1, 2], dtype=np.int64)
    cm = _confusion({1: {1: 2, 2: 1}, 2: {0: 1}})
    row = eval_volume(pred,
                      label,
                      4,
                      25,
                      0, ("dice", ),
                      rib_metrics=True,
                      conf_mat=cm)
    assert row["confusion"] == cm.tolist()
    assert row["fg_purity_cw"][1] == 2 / 3
    assert row["fg_recognition_coverage_cw"][2] == 0.0

    summary = reduce_records({"volume": row},
                             cm,
                             25,
                             0, ("dice", ),
                             rib_metrics=True)
    assert summary["fg_scored_pair_count"] == 1
    assert summary["fg_unrecognized_pair_count"] == 1

    binary = eval_volume(np.array([0, 1]),
                         np.array([0, 1]),
                         2,
                         2,
                         0, ("dice", ),
                         rib_metrics=True)
    assert "fg_purity_cw" not in binary


def test_foreground_confusion_metrics_validates_ribseg_shape():
    with np.testing.assert_raises_regex(ValueError, "25x25"):
        foreground_confusion_metrics(np.zeros((2, 2), dtype=np.int64))


def test_foreground_confusion_rejects_invalid_counts_and_empty_summary_is_unscored(
):
    for value in (-1, .5, float('nan'), float('inf')):
        cm = np.zeros((25, 25))
        cm[1, 1] = value
        with np.testing.assert_raises(ValueError):
            foreground_confusion_metrics(cm)
    summary = summarize_foreground_confusion(
        {'v': foreground_confusion_metrics(_confusion({1: {
            0: 2
        }}))})
    assert summary['fg_scored_pair_count'] == 0
    assert summary['fg_purity_mean'] is None
    assert summary['fg_fragment_gt1_fraction'] is None
    assert summary['fg_unrecognized_pair_count'] == 1
    assert summary['fg_recognition_coverage_mean'] == 0


def test_purity_tail_filters_individual_pairs_strictly_before_aggregation():
    records = {
        'a': foreground_confusion_metrics(_confusion({
            1: {1: 3, 2: 2},  # .6
            2: {0: 5},        # unscored
            3: {3: 7, 4: 3},  # .7: excluded at the exact threshold
        })),
        'b': foreground_confusion_metrics(_confusion({
            1: {1: 1, 2: 1},  # .5, same rib in a different volume
            3: {3: 8, 4: 2},  # .8
        })),
    }
    summary = summarize_foreground_confusion(records, purity_threshold=.7)
    assert summary['fg_purity_tail_threshold'] == .7
    assert summary['fg_purity_tail_count'] == 2
    assert summary['fg_purity_tail_median'] == .55
    assert summary['fg_purity_tail_max'] == .6
    assert summary['fg_purity_tail_min'] == .5
    assert summary['fg_scored_pair_count'] == 4
    assert summary['fg_purity_max'] == .8  # unfiltered keys stay unchanged

    wider = summarize_foreground_confusion(records)
    assert wider['fg_purity_tail_threshold'] == .9
    assert wider['fg_purity_tail_count'] == 4
    assert np.isclose(wider['fg_purity_tail_median'], .65)
    empty = summarize_foreground_confusion(records, purity_threshold=.5)
    assert empty['fg_purity_tail_count'] == 0
    for stat in ('median', 'max', 'min'):
        assert empty['fg_purity_tail_' + stat] is None
    for threshold in (-.1, 1.1, float('nan'), float('inf')):
        with np.testing.assert_raises(ValueError):
            summarize_foreground_confusion(records, purity_threshold=threshold)
