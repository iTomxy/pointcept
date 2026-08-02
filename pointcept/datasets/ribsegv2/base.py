import json


IGNORE_VOLUMES = (452, 485, 490)
IGNORE_VOLUMES += (
    # (23 Apr 2026, iTom) potential wrong label
    344, 439, 462, 471, 487, 540, 652, 653,
    # (22 Jan 2026, iTom) too noisy, even binary seg cannot work well, ignore for now
    # 501, 507, 570, 589, 630, 653,
)
SPLITS = {
    "train": [x for x in range(1, 421) if x not in IGNORE_VOLUMES],
    "val": [x for x in range(421, 501) if x not in IGNORE_VOLUMES],
    "test": [x for x in range(501, 661) if x not in IGNORE_VOLUMES],
}
SPLITS["all"] = SPLITS["train"] + SPLITS["val"] + SPLITS["test"]
# MIN_HU = 200
INCOMPLETE_VOLS = set()
with open("data/ribsegv2/ribsegv2-statistics.json", "r") as f:
    for line in f:
        d = json.loads(line)
        if "vid" in d and len(d["class_set"]) < 24 + 1:
            INCOMPLETE_VOLS.add(int(d["vid"]))
