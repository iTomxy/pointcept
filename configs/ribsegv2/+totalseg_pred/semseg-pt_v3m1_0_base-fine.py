"""Phase 2: tuned PTv3 on the 62-class fine complemented label space."""

_base_ = ["_ptv3_base.py"]

save_path = "exp/ribsegv2/+totalseg_pred/semseg-pt_v3m1_0_base-fine"

fine_class_names = (
    ("background",)
    + tuple("rib_left_{}".format(i) for i in range(1, 13))
    + tuple("rib_right_{}".format(i) for i in range(1, 13))
    + ("sacrum", "vertebrae_S1")
    + tuple("vertebrae_L{}".format(i) for i in range(5, 0, -1))
    + tuple("vertebrae_T{}".format(i) for i in range(12, 0, -1))
    + tuple("vertebrae_C{}".format(i) for i in range(7, 0, -1))
    + (
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
)

model = dict(num_classes=62)

data = dict(
    num_classes=62,
    names=fine_class_names,
    train=dict(bone_label_granularity="fine"),
    val=dict(bone_label_granularity="fine"),
    test=dict(bone_label_granularity="fine"),
)

