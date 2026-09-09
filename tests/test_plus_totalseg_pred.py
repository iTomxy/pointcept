import stat

import nibabel as nib
import numpy as np
import pytest

from pointcept.datasets.ribsegv2.plus_totalseg_pred import (
    COARSE_CLASS_NAMES,
    FINE_CLASS_NAMES,
    NUM_COARSE_CLASSES,
    NUM_FINE_CLASSES,
    TOTALSEG_TO_COARSE,
    TOTALSEG_TO_FINE,
    CombineBoneLabel,
    combine_bone_label,
    patch_totalseg_pred,
    summarise_totalseg_classes,
)
from pointcept.datasets.ribsegv2.preproc import _load_aligned_volume
from pointcept.datasets.totalseg import TOTALSEG_CLASSES, TOTALSEG_RIB_CLASSES


def test_class_spaces_and_rib_ordering():
    assert NUM_COARSE_CLASSES == 10
    assert NUM_FINE_CLASSES == 62
    assert COARSE_CLASS_NAMES[0] == "background"
    assert list(FINE_CLASS_NAMES[1:13]) == [f"rib_left_{i}" for i in range(1, 13)]
    assert list(FINE_CLASS_NAMES[13:25]) == [f"rib_right_{i}" for i in range(1, 13)]
    assert FINE_CLASS_NAMES[25:32] == (
        "sacrum",
        "vertebrae_S1",
        "vertebrae_L5",
        "vertebrae_L4",
        "vertebrae_L3",
        "vertebrae_L2",
        "vertebrae_L1",
    )
    assert FINE_CLASS_NAMES[51:] == (
        "scapula_left",
        "scapula_right",
        "humerus_left",
        "humerus_right",
        "clavicula_left",
        "clavicula_right",
        "sternum",
        "costal_cartilages",
        "skull",
        "hip_left",
        "hip_right",
    )


def test_coarse_and_fine_manual_precedence():
    """RibSegv2's manual label wins wherever it exists; the first point has no
    manual label and a TotalSegmentator rib prediction (92, rib_left_1), which
    is discarded rather than filling in as rib -- TotalSegmentator's rib ids
    are exclusively RibSegv2's, so an unlabelled point stays background."""
    source = {
        "segment": np.array([0, 3, 0, 24], dtype=np.int64),
        "totalseg_pred": np.array([92, 3, 25, 115], dtype=np.int64),
    }
    coarse = combine_bone_label(dict(source), drop_source=False)["segment"]
    fine = combine_bone_label(dict(source), granularity="fine", drop_source=False)["segment"]
    assert coarse.tolist() == [0, 1, 2, 1]
    assert fine.tolist() == [0, 3, 25, 24]


def test_totalseg_rib_ids_map_to_background_in_both_tables():
    """The rib class must be unreachable through the TotalSegmentator lookup."""
    for totalseg_id in TOTALSEG_RIB_CLASSES:
        assert TOTALSEG_TO_COARSE[totalseg_id] == 0
        assert TOTALSEG_TO_FINE[totalseg_id] == 0


def _rib_totalseg_id() -> int:
    return TOTALSEG_RIB_CLASSES[0]


def test_coarse_combine_discards_a_totalseg_rib_id_ribsegv2_never_labelled():
    rib_id = _rib_totalseg_id()
    data = {
        "segment": np.array([0], dtype=np.int64),
        "totalseg_pred": np.array([rib_id], dtype=np.int64),
    }
    combined = combine_bone_label(data, granularity="coarse")
    assert combined["segment"][0] == COARSE_CLASS_NAMES.index("background")


def test_coarse_combine_keeps_the_ribsegv2_rib_override():
    rib_id = _rib_totalseg_id()
    data = {
        "segment": np.array([7], dtype=np.int64),
        "totalseg_pred": np.array([rib_id], dtype=np.int64),
    }
    combined = combine_bone_label(data, granularity="coarse")
    assert combined["segment"][0] == COARSE_CLASS_NAMES.index("rib")


def test_fine_combine_discards_a_totalseg_rib_id_ribsegv2_never_labelled():
    rib_id = _rib_totalseg_id()
    data = {
        "segment": np.array([0], dtype=np.int64),
        "totalseg_pred": np.array([rib_id], dtype=np.int64),
    }
    combined = combine_bone_label(data, granularity="fine")
    assert combined["segment"][0] == 0


def test_fine_combine_keeps_the_ribsegv2_rib_override():
    rib_id = _rib_totalseg_id()
    data = {
        "segment": np.array([7], dtype=np.int64),
        "totalseg_pred": np.array([rib_id], dtype=np.int64),
    }
    combined = combine_bone_label(data, granularity="fine")
    assert combined["segment"][0] == 7


def test_non_rib_bones_are_unaffected():
    vertebrae_id = next(id_ for id_, name in TOTALSEG_CLASSES.items() if name == "vertebrae_T12")
    sternum_id = next(id_ for id_, name in TOTALSEG_CLASSES.items() if name == "sternum")

    coarse = combine_bone_label(
        {
            "segment": np.array([0, 0], dtype=np.int64),
            "totalseg_pred": np.array([vertebrae_id, sternum_id], dtype=np.int64),
        },
        granularity="coarse",
    )
    assert coarse["segment"][0] == COARSE_CLASS_NAMES.index("vertebrae")
    assert coarse["segment"][1] == COARSE_CLASS_NAMES.index("sternum")

    fine = combine_bone_label(
        {
            "segment": np.array([0, 0], dtype=np.int64),
            "totalseg_pred": np.array([vertebrae_id, sternum_id], dtype=np.int64),
        },
        granularity="fine",
    )
    assert FINE_CLASS_NAMES[fine["segment"][0]] == "vertebrae_T12"
    assert FINE_CLASS_NAMES[fine["segment"][1]] == "sternum"


def test_excluded_and_invalid_totalseg_ids_are_background():
    data = {
        "segment": np.zeros(4, dtype=np.int64),
        "totalseg_pred": np.array([88, -1, 999, 1], dtype=np.int64),
    }
    result = combine_bone_label(data, drop_source=False)["segment"]
    assert result.tolist() == [0, 0, 0, 0]


def test_source_deletion_and_retention():
    data = {"segment": np.array([0]), "totalseg_pred": np.array([92])}
    combine_bone_label(data)
    assert "totalseg_pred" not in data
    data = {"segment": np.array([0]), "totalseg_pred": np.array([92])}
    combine_bone_label(data, drop_source=False)
    assert "totalseg_pred" in data


def test_transform_metadata_driven_fine_mode():
    # segment=24 (a RibSegv2 label) is required here: TotalSegmentator's rib
    # predictions are discarded outright, so a bare totalseg_pred no longer
    # distinguishes fine mode from coarse (both would now yield background).
    # With the RibSegv2 override in effect, fine mode copies the label itself
    # (24) while coarse mode would instead collapse it to the single rib id (1).
    data = {
        "segment": np.array([24]),
        "totalseg_pred": np.array([115]),
        "bone_label_granularity": "fine",
    }
    result = CombineBoneLabel(granularity=None)(data)
    assert result["segment"].tolist() == [24]
    assert "bone_label_granularity" not in result


@pytest.mark.parametrize(
    "kwargs",
    [
        {"granularity": "bad"},
        {"label_key": "missing"},
        {"totalseg_key": "missing"},
    ],
)
def test_invalid_arguments(kwargs):
    data = {"segment": np.array([0]), "totalseg_pred": np.array([0])}
    with pytest.raises((ValueError, KeyError)):
        combine_bone_label(data, **kwargs)


def test_shape_and_dtype_errors():
    with pytest.raises(ValueError):
        combine_bone_label({"segment": np.zeros((1, 1)), "totalseg_pred": np.zeros(1)})
    with pytest.raises(TypeError, match="totalseg_pred must have an integer dtype"):
        combine_bone_label({"segment": np.array([0]), "totalseg_pred": np.array([1.5])})
    with pytest.raises(TypeError, match="segment must have an integer dtype"):
        combine_bone_label({"segment": np.array([0.0]), "totalseg_pred": np.array([1])})


def test_manual_labels_outside_ribsegv2_range_error():
    with pytest.raises(ValueError):
        combine_bone_label({"segment": np.array([25]), "totalseg_pred": np.array([0])})


def test_patcher_rejects_a_different_totalseg_subtask_before_io():
    with pytest.raises(ValueError, match="subtask must be 'total'"):
        patch_totalseg_pred(subtask="total_mr")


def test_summary_reads_cached_point_labels(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    np.savez(cache / "1.npz", totalseg_pred=np.array([0, 25, 92, 92]))
    np.savez(cache / "2.npz", totalseg_pred=np.array([0, 116]))
    log = tmp_path / "summary.jsonl"

    summary = summarise_totalseg_classes(cache, log)

    assert summary["n_volumes"] == 2
    assert summary["n_points"] == 6
    by_id = {record["id"]: record for record in summary["classes"]}
    assert by_id[92]["name"] == "rib_left_1"
    assert by_id[92]["n_points"] == 2
    assert by_id[92]["n_volumes"] == 1
    assert not (log.stat().st_mode & stat.S_IWUSR)


def test_aligned_volume_reorients_from_its_own_affine(tmp_path):
    array = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    path = tmp_path / "ras.nii.gz"
    nib.save(nib.Nifti1Image(array, np.eye(4)), path)
    lps_affine = np.array(
        [
            [-1.0, 0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    aligned = _load_aligned_volume(
        path, lps_affine, array.shape, 491, dtype=np.int16, what="test prediction"
    )

    assert np.array_equal(aligned, array[::-1, ::-1, :])
    with pytest.raises(ValueError, match="reference grid"):
        _load_aligned_volume(path, lps_affine, (9, 9, 9), 491)
    with pytest.raises(ValueError, match="does not sit on the same grid"):
        _load_aligned_volume(path, np.eye(4), array.shape, 491)
